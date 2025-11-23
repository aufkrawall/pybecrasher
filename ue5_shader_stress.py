# MIT License
# Copyright (c) 2024 aufkrawall
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import subprocess
import random
import time
import argparse
import shutil
import datetime
import urllib.request
import zipfile
import io
import threading
import ctypes
import zlib
import multiprocessing
import signal
import hashlib
import lzma
import platform
from ctypes import wintypes

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

VERSION = "13.0"

DXC_CONFIG = {
    "URL": "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2407/dxc_2024_07_31.zip",
    "DIR": "dxc_bin",
    "TEMP": "temp_shaders"
}

LOG_FILE = "pybecrasher.log"
SYSTEM_CORES = os.cpu_count() or 1
CYCLE_DURATION = 60
MAX_NOISE_THREADS = 4 # (1 RAM, 1 IO, 2 Integrity)

CRASH_CODES = {
    -1073741819: "Access Violation (0xC0000005) - RAM/Vcore Unstable",
    3221225477: "Access Violation (0xC0000005) - RAM/Vcore Unstable",
    -1073741571: "Stack Overflow (0xC00000FD)",
    3221225725: "Stack Overflow (0xC00000FD)",
    -1073740791: "Stack Buffer Overrun (0xC0000409)",
    3221226505: "Stack Buffer Overrun (0xC0000409)",
    -1073741676: "Integer Divide by Zero (0xC0000094)",
    3221225620: "Integer Divide by Zero (0xC0000094)",
    0xC0000428:  "Integrity Checksum Failure (CPU/RAM Calc Error)",
}

class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"

# ==============================================================================
# UTILITIES
# ==============================================================================

class SysUtils:
    @staticmethod
    def set_timer_resolution():
        try: ctypes.windll.winmm.timeBeginPeriod(1)
        except: pass

    @staticmethod
    def get_total_ram():
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys

    @staticmethod
    def get_resource_path(filename):
        if os.path.exists(filename):
            return os.path.abspath(filename)
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, filename)
        return None

    @staticmethod
    def print_file_content(filename, title):
        path = SysUtils.get_resource_path(filename)
        SysUtils.clear_screen()
        print(f"{Colors.HEADER}=== {title} ==={Colors.ENDC}\n")
        try:
            if path and os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    print(f.read())
            else:
                print(f"{Colors.FAIL}File '{filename}' not found.{Colors.ENDC}")
        except Exception as e:
            print(f"Error reading file: {e}")
        print(f"\n{Colors.OKBLUE}{'='*40}{Colors.ENDC}")
        input("\nPress Enter to return...")

    @staticmethod
    def clear_screen():
        os.system("cls" if os.name == "nt" else "clear")

# ==============================================================================
# LOGGING SYSTEM
# ==============================================================================

class CrashLogger:
    @staticmethod
    def start_log(mode, threads):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"=== PyBeCrasher v{VERSION} Session Log ===\n")
                f.write(f"Start Time: {datetime.datetime.now()}\n")
                f.write(f"System: {platform.system()} {platform.release()} ({platform.machine()})\n")
                f.write(f"CPU Cores: {SYSTEM_CORES}\n")
                f.write(f"Mode: {mode}\n")
                f.write(f"Thread Config: {threads}\n")
                f.write("-------------------------------------------\n")
                f.write("Status: RUNNING\n")
        except: pass

    @staticmethod
    def log_failure(code, reason, context):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n!!! HARDWARE FAILURE DETECTED at {datetime.datetime.now()} !!!\n")
                f.write(f"Exit Code: {code} ({hex(code & 0xFFFFFFFF)})\n")
                f.write(f"Reason:    {reason}\n")
                f.write(f"Context:   {context}\n")
        except: pass

    @staticmethod
    def cleanup_log():
        if os.path.exists(LOG_FILE):
            try: os.remove(LOG_FILE)
            except: pass

# ==============================================================================
# ASSET GENERATION (DXC & SHADERS)
# ==============================================================================

class AssetManager:
    SHADER_RAM = """
    struct PS_INPUT {{ float4 Pos : SV_POSITION; float2 UV : TEXCOORD; }};
    // Procedural pseudo-random generator
    float4 get_data(uint idx) {{
        uint h = idx * 0xdeadbeef;
        h = (h ^ (h >> 4)) * 0x1234567;
        float v = float(h & 0xffff) / 65536.0;
        return float4(v, v*0.5, 0, 1);
    }}
    float4 PSMain(PS_INPUT i) : SV_TARGET {{
        uint idx = (uint)(abs(i.Pos.x)*100000.0) % 16384;
        float4 acc = 0;
        [loop] for(int k=0; k<{steps}; k++) {{
            acc = mad(acc, get_data((idx + k*127) % 16384), 0.0001);
        }}
        return acc;
    }}"""

    SHADER_HYBRID = """
    #define MAX_STEPS {steps}
    struct PS_INPUT {{ float4 Pos : SV_POSITION; float2 UV : TEXCOORD; }};
    float4 PSMain(PS_INPUT input) : SV_TARGET {{
        float4 f = float4(input.UV, 1.0, 0.5);
        uint4 i = uint4(1, 2, 3, 4);
        [unroll({unroll})] for(int k=0; k<MAX_STEPS; k++) {{
            f = mad(f, 0.123, 0.456);
            i = (i << 1) ^ (i >> 1) ^ 0xA5A5A5A5;
            if((k&127)==0) f += sin(f);
        }}
        return f + (float4)i;
    }}"""

    @staticmethod
    def get_dxc_binary():
        bin_path = os.path.abspath(os.path.join(DXC_CONFIG["DIR"], "bin", "dxc.exe"))
        if os.path.exists(bin_path): return bin_path

        print(f"{Colors.WARNING}Downloading DirectXShaderCompiler from Microsoft GitHub...{Colors.ENDC}")
        try:
            if not os.path.exists(DXC_CONFIG["DIR"]): os.makedirs(DXC_CONFIG["DIR"])
            req = urllib.request.Request(DXC_CONFIG["URL"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r:
                with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                    for f in z.namelist():
                        if f.startswith("bin/x64/"): z.extract(f, DXC_CONFIG["DIR"])

            src = os.path.join(DXC_CONFIG["DIR"], "bin", "x64")
            dst = os.path.join(DXC_CONFIG["DIR"], "bin")
            if os.path.exists(src):
                for f in os.listdir(src): shutil.move(os.path.join(src, f), dst)
                os.rmdir(src)
            return bin_path
        except Exception as e:
            print(f"{Colors.FAIL}Download Failed: {e}{Colors.ENDC}")
            sys.exit(1)

    @staticmethod
    def setup_workspace():
        if os.path.exists(DXC_CONFIG["TEMP"]): shutil.rmtree(DXC_CONFIG["TEMP"])
        os.makedirs(DXC_CONFIG["TEMP"])

    @staticmethod
    def generate_shader_code(rng):
        if rng.random() < 0.4:
            return AssetManager.SHADER_RAM.format(steps=rng.randint(60000, 100000))
        else:
            return AssetManager.SHADER_HYBRID.format(steps=rng.randint(600000, 1000000), unroll=1024)

    @staticmethod
    def precompute_benchmark_pool(count):
        sys.stdout.write(f"Pre-generating {count} static benchmark shaders... ")
        pool = []
        rng = random.Random(42) # Deterministic Seed
        for i in range(count):
            fname = os.path.join(DXC_CONFIG["TEMP"], f"bench_{i}.hlsl")
            code = AssetManager.generate_shader_code(rng)
            with open(fname, "w", encoding="utf-8") as f: f.write(code)
            pool.append(fname)
        print(f"{Colors.OKGREEN}Done.{Colors.ENDC}")
        return pool

# ==============================================================================
# NOISE SUB-SYSTEM (Running in separate process)
# ==============================================================================

class NoiseModules:
    class RamAnvil:
        def __init__(self, stop, trigger): self.stop, self.trigger = stop, trigger
        def run(self):
            try:
                # 60% Allocation
                sz = int(SysUtils.get_total_ram() * 0.60)
                buf = bytearray(sz)
                view = memoryview(buf)
                max_idx = sz - 4096
                for i in range(0, sz, 409600): buf[i] = 1
            except: return

            rng = random.Random()
            while not self.stop.is_set():
                if not self.trigger.is_set(): time.sleep(0.1); continue
                try:
                    mode = rng.randint(0, 10)
                    idx = rng.randint(0, max_idx)
                    if mode < 6: view[idx] = rng.randint(0, 255)
                    elif mode < 9:
                        base = (idx // 4096) * 4096
                        view[base] ^= 0xFF
                    else:
                        end = min(idx + 4096, sz)
                        for i in range(idx, end, 64):
                            view[i] = (view[i] + 1) % 256
                except: pass

    class IntegrityStress:
        def __init__(self, stop, q, trigger): self.stop, self.q, self.trigger = stop, q, trigger
        def run(self):
            raw = os.urandom(1024 * 16384)
            hashes = {
                'crc': zlib.crc32(raw) & 0xFFFFFFFF,
                'blake': hashlib.blake2b(raw, digest_size=32).digest()
            }
            blobs = {
                'zlib': zlib.compress(raw, level=6),
                'lzma': lzma.compress(raw, preset=6)
            }

            while not self.stop.is_set():
                if not self.trigger.is_set(): time.sleep(0.1); continue
                try:
                    for alg, blob in blobs.items():
                        data = zlib.decompress(blob) if alg == 'zlib' else lzma.decompress(blob)

                        if (zlib.crc32(data) & 0xFFFFFFFF) != hashes['crc']:
                            self.q.put((0xC0000428, "CRC FAILURE", f"{alg.upper()} Decomp Mismatch"))
                            self.stop.set(); return

                        if hashlib.blake2b(data, digest_size=32).digest() != hashes['blake']:
                            self.q.put((0xC0000428, "HASH FAILURE", f"{alg.upper()} Blake2b Mismatch"))
                            self.stop.set(); return
                except Exception: pass

    class IoStress:
        def __init__(self, stop, folder, trigger):
            self.stop, self.folder, self.trigger = stop, folder, trigger
            self.f = os.path.join(folder, "io.dat")

        def run(self):
            try:
                if not os.path.exists(self.folder): os.makedirs(self.folder, exist_ok=True)
            except: pass

            if not os.path.exists(self.f):
                try:
                    with open(self.f, "wb") as f: f.write(os.urandom(1024*1024*1024))
                except: return

            k32 = ctypes.windll.kernel32
            h = k32.CreateFileW(self.f, 0x80000000, 0, None, 3, 0x20000000, None)
            if h == -1: return

            buf = ctypes.create_string_buffer(4096 + 4095)
            aligned_buf = (ctypes.addressof(buf) + 4095) & ~4095
            rng = random.Random()
            max_pos = (1024*1024*1024 - 4096) // 4096
            read_bytes = wintypes.DWORD()

            try:
                while not self.stop.is_set():
                    if not self.trigger.is_set(): time.sleep(0.1); continue
                    k32.SetFilePointer(h, rng.randint(0, max_pos) * 4096, None, 0)
                    k32.ReadFile(h, ctypes.c_void_p(aligned_buf), 4096, ctypes.byref(read_bytes), None)
            finally: k32.CloseHandle(h)

def noise_process_entry(stop_evt, msg_queue, temp_dir, active_evt, thread_limit):
    """Entry point for the isolated noise process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    local_io_dir = os.path.join(temp_dir, "io_stress")

    modules = [
        NoiseModules.IntegrityStress(stop_evt, msg_queue, active_evt),
        NoiseModules.RamAnvil(stop_evt, active_evt),
        NoiseModules.IntegrityStress(stop_evt, msg_queue, active_evt),
        NoiseModules.IoStress(stop_evt, local_io_dir, active_evt)
    ]

    threads = []
    for mod in modules[:thread_limit]:
        t = threading.Thread(target=mod.run, daemon=True)
        t.start()
        threads.append(t)

    while not stop_evt.is_set(): time.sleep(1)

    if os.path.exists(local_io_dir):
        try: shutil.rmtree(local_io_dir, ignore_errors=True)
        except: pass

# ==============================================================================
# MAIN STRESS CONTROLLER
# ==============================================================================

class StressController:
    def __init__(self, args):
        self.args = args
        self.dxc = AssetManager.get_dxc_binary()
        self.crashed_state = False

        self.noise_count = 0 if args.mode == "benchmark" else min(MAX_NOISE_THREADS, max(0, SYSTEM_CORES - 1))

        self.comp_count = SYSTEM_CORES
        if args.mode != "benchmark":
            self.comp_count = max(1, SYSTEM_CORES - 3)

        self.total_threads_display = f"{self.comp_count} (Compilers) + {self.noise_count} (Noise)"

        CrashLogger.start_log(args.mode, self.total_threads_display)

        self.stats = {'act': 0, 'ok': 0, 'err': 0}
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(self.comp_count) if args.mode == "variable" else None
        self.stop = threading.Event()
        self.mp_stop = multiprocessing.Event()
        self.mp_queue = multiprocessing.Queue()
        self.mp_trigger = multiprocessing.Event()
        self.mp_trigger.set()
        self.diag_printed = False

    def run(self):
        print(f"{Colors.HEADER}=== PyBeCrasher v{VERSION} ==={Colors.ENDC}")
        print(f"Mode: {self.args.mode} | Load: {self.total_threads_display} | Priority: BELOW_NORMAL")

        AssetManager.setup_workspace()

        # Pre-generate files ONLY for benchmark mode to avoid disk writes during test
        bench_pool = []
        if self.args.mode == "benchmark":
            bench_pool = AssetManager.precompute_benchmark_pool(max(200, self.comp_count * 4))

        noise_proc = None
        if self.noise_count > 0:
            noise_proc = multiprocessing.Process(
                target=noise_process_entry,
                args=(self.mp_stop, self.mp_queue, DXC_CONFIG["TEMP"], self.mp_trigger, self.noise_count)
            )
            noise_proc.daemon = True
            noise_proc.start()

        threads = []
        self.target_active = self.comp_count if self.args.mode != "variable" else 0

        for i in range(self.comp_count):
            t = threading.Thread(target=self.compiler_worker, args=(i, bench_pool))
            t.daemon = True
            t.start()
            threads.append(t)

        start_time = time.time()
        try:
            self._monitor_loop(start_time, noise_proc)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup(threads, noise_proc)

    def _monitor_loop(self, start_time, noise_proc):
        bench_data = {'start': start_time, 'cc': 0, 'scores': []}

        while not self.stop.is_set():
            now = time.time()
            elapsed = now - start_time

            if not self.mp_queue.empty():
                code, reason, msg = self.mp_queue.get()
                self.print_crash(code, reason, msg)
                return

            tgt, phase_name, noise_active = self.get_phase(elapsed)

            if self.args.mode == "variable":
                with self.lock: self.target_active = tgt

            if noise_proc:
                if noise_active: self.mp_trigger.set()
                else: self.mp_trigger.clear()

            self.update_ui(elapsed, phase_name, noise_active, bench_data)

            if self.args.duration and elapsed > self.args.duration: return
            time.sleep(0.05)

    def get_phase(self, elapsed):
        if self.args.mode != "variable": return self.comp_count, "STEADY MAX", True

        t = elapsed % CYCLE_DURATION
        if t < 20: return self.comp_count, "HEAT SOAK", True
        if t < 35: return (0 if int(t)%2==0 else self.comp_count), "SPIKE WAVE", True
        if t < 50:
            random.seed(int(t*10))
            opt = random.choice([(0,"CHAOS 0%"), (max(1,self.comp_count//2),"CHAOS 50%"), (self.comp_count,"CHAOS 100%")])
            return opt[0], opt[1], True
        return (1 if int(t)%2==0 else 2), "BOOST (1T)", False

    def update_ui(self, elapsed, phase, noise_on, bench):
        with self.lock:
            a, c, e = self.stats['act'], self.stats['ok'], self.stats['err']

        rate = 0.0
        if elapsed > 1.0: rate = c / elapsed

        extra = ""
        if self.args.mode == "benchmark":
            if time.time() - bench['start'] > 60:
                score = (c - bench['cc']) / 60.0
                bench['scores'].append(score)
                bench['start'] = time.time()
                bench['cc'] = c
            best = max(bench['scores']) if bench['scores'] else 0
            extra = f" | Best 60s: {best:.1f}/s"

        noise_disp = self.noise_count if (noise_on and self.noise_count > 0) else 0

        try: cols = shutil.get_terminal_size().columns
        except: cols = 80

        status = (f"Time: {str(datetime.timedelta(seconds=int(elapsed)))} | "
                  f"Rate: {rate:.1f}/s{extra} | "
                  f"Act: {a + noise_disp} ({a}C+{noise_disp}N) | Err: {e} | "
                  f"{Colors.OKGREEN if 'IDLE' in phase else Colors.WARNING}{phase}{Colors.ENDC}")

        print(f"\r{status:<{cols}}", end="", flush=True)

    def compiler_worker(self, thread_id, bench_pool):
        rng = random.Random()
        if self.args.mode == "benchmark":
            rng.seed(42 + thread_id)
        else:
            rng.seed(time.time() + thread_id)

        flags = 0x00004000 | 0x00000008 if os.name == 'nt' else 0
        my_shader_file = os.path.join(DXC_CONFIG["TEMP"], f"worker_{thread_id}.hlsl")

        while not self.stop.is_set():
            if self.args.mode == "variable":
                if self.barrier and rng.random() < 0.05:
                    try: self.barrier.wait(timeout=0.1)
                    except: pass

                wait = False
                with self.lock:
                    if self.stats['act'] >= self.target_active: wait = True
                    else: self.stats['act'] += 1

                if wait:
                    for _ in range(5000): pass
                    continue
            else:
                with self.lock: self.stats['act'] += 1

            try:
                target_file = ""
                if self.args.mode == "benchmark":
                    target_file = rng.choice(bench_pool)
                else:
                    target_file = my_shader_file
                    code = AssetManager.generate_shader_code(rng)
                    with open(target_file, "w", encoding="utf-8") as f: f.write(code)

                res = subprocess.run(
                    [self.dxc, "-T", "ps_6_6", "-O3", "-Vd", "-E", "PSMain", "-HV", "2021", "-all_resources_bound", target_file, "-Fo", "NUL"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, creationflags=flags
                )

                code = res.returncode
                with self.lock:
                    self.stats['act'] -= 1
                    if code == 0: self.stats['ok'] += 1
                    else:
                        if code in CRASH_CODES or abs(code) > 1000000:
                            self.mp_queue.put((code, CRASH_CODES.get(code, "CRASH"), "Compiler Process Died"))
                            self.stop.set()
                        else:
                            self.stats['err'] += 1
                            if not self.diag_printed:
                                print(f"\n{Colors.WARNING}[DIAGNOSTIC] DXC Failed: {res.stderr}{Colors.ENDC}")
                                self.diag_printed = True
            except Exception as e:
                with self.lock: self.stats['act'] -= 1

    def print_crash(self, code, reason, msg):
        self.crashed_state = True
        CrashLogger.log_failure(code, reason, msg)
        print(f"\n\n{Colors.FAIL}{'='*60}\nHARDWARE FAILURE DETECTED\n{'='*60}{Colors.ENDC}")
        print(f"Exit Code: {code}\nAnalysis:  {reason}\nContext:   {msg}\n")
        print(f"{Colors.WARNING}Log saved to: {os.path.abspath(LOG_FILE)}{Colors.ENDC}")
        self.stop.set()

    def cleanup(self, threads, noise_proc):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.stop.set()

        try: self.mp_stop.set()
        except: pass

        if noise_proc:
            try:
                if noise_proc.is_alive():
                    noise_proc.terminate()
                    noise_proc.join(timeout=0.5)
                if noise_proc.is_alive():
                    noise_proc.kill()
            except: pass

        if not self.crashed_state:
            CrashLogger.cleanup_log()

        SysUtils.set_timer_resolution()
        if os.path.exists(DXC_CONFIG["TEMP"]): shutil.rmtree(DXC_CONFIG["TEMP"], ignore_errors=True)
        print("\nClean exit.")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main_menu():
    SysUtils.clear_screen()
    print(f"{Colors.HEADER}========================================================{Colors.ENDC}")
    print(f"{Colors.HEADER}    PYBECRASHER v{VERSION} - UE5 STRESS SIMULATOR       {Colors.ENDC}")
    print(f"{Colors.HEADER}========================================================{Colors.ENDC}")
    print(" 1. Variable Load (Chaos)  [Recommended for Instability]")
    print(" 2. Steady Load            [Thermal Soak]")
    print(" 3. Benchmark              [Throughput Score]")
    print(" 4. View README")
    print(" 5. View LICENSE")
    print("--------------------------------------------------------")

    choice = input(" Select Mode [1]: ").strip()
    mode = "variable"
    if choice == "2": mode = "steady"
    if choice == "3": mode = "benchmark"
    if choice == "4":
        SysUtils.print_file_content("README.md", "README")
        return main_menu()
    if choice == "5":
        SysUtils.print_file_content("LICENSE", "LICENSE")
        return main_menu()

    if getattr(sys, 'frozen', False): exe = sys.executable
    else: exe = sys.executable + " " + os.path.abspath(__file__)

    cmd = f'{exe} --worker --mode {mode}'
    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        pass

    print("\nTest finished.")
    try: input("Press Enter to close...")
    except: pass

if __name__ == "__main__":
    multiprocessing.freeze_support()
    SysUtils.set_timer_resolution()

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", default="variable")
    parser.add_argument("--duration", type=int, default=0)
    args = parser.parse_args()

    try:
        if args.worker:
            ctrl = StressController(args)
            ctrl.run()
        else:
            main_menu()
    except KeyboardInterrupt:
        pass # Clean exit on Ctrl+C
