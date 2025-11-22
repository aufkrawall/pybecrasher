# MIT License
#
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
import collections
import signal
import hashlib
import lzma
from ctypes import wintypes

# --- Configuration ---
DXC_DOWNLOAD_URL = (
    "https://github.com/microsoft/DirectXShaderCompiler/releases/"
    "download/v1.8.2407/dxc_2024_07_31.zip"
)
DXC_LOCAL_DIR = "dxc_bin"
TEMP_DIR = "temp_shaders"

LOGICAL_CORES = os.cpu_count()
# Default to -1 to indicate "Auto-tuning" based on mode
DEFAULT_WORKERS = -1
CYCLE_DURATION = 60

# Pulse configuration for Chaos mode barrier sync
PULSE_INTERVAL = 5.0
PULSE_DURATION = 0.5

# Maximum noise threads to attempt spawning (1 Ram, 1 Cache, 2 Integrity, 1 IO)
MAX_NOISE_THREADS = 5

CRASH_EXIT_CODES = {
    -1073741819: "Access Violation (0xC0000005) - RAM/Vcore Unstable",
    3221225477: "Access Violation (0xC0000005) - RAM/Vcore Unstable",
    -1073741571: "Stack Overflow (0xC00000FD)",
    3221225725: "Stack Overflow (0xC00000FD)",
    -1073740791: "Stack Buffer Overrun (0xC0000409)",
    3221226505: "Stack Buffer Overrun (0xC0000409)",
    -1073741676: "Integer Divide by Zero (0xC0000094)",
    3221225620: "Integer Divide by Zero (0xC0000094)",
    0xC0000428: "Integrity Checksum Failure (Core/RAM Calculation Error)",
}

class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


# --- RESOURCE HELPER ---

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def print_file_content(filename, title):
    path = get_resource_path(filename)
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{Colors.HEADER}=== {title} ==={Colors.ENDC}\n")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            print(
                f"{Colors.FAIL}File '{filename}' not found in bundle."
                f"{Colors.ENDC}"
            )
    except Exception as e:
        print(f"Error reading file: {e}")
    print(f"\n{Colors.OKBLUE}========================================{Colors.ENDC}")
    try:
        input("\nPress Enter to return to menu...")
    except KeyboardInterrupt:
        # Menu loop handles this
        pass


# --- SYSTEM TUNING (Windows timer resolution) ---

try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_total_ram():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys


# --- SHADER TEMPLATES ---

# Hybrid ALU + integer bit-twiddling workload
SHADER_TEMPLATE_HYBRID = """
#define MAX_STEPS {steps}
struct PS_INPUT {{ float4 Pos : SV_POSITION; float2 UV : TEXCOORD; }};
float4 PSMain(PS_INPUT input) : SV_TARGET {{
    float4 f0 = float4(input.UV, 1.0, 0.5);
    float4 f1 = float4(input.UV.yx, 0.5, 1.0);
    float4 f2 = float4(0.1, 0.2, 0.3, 0.4);
    float4 f3 = float4(0.9, 0.8, 0.7, 0.6);
    float4 c1 = float4(0.123, 0.456, 0.789, 1.234);

    uint4 i0 = uint4(1, 2, 3, 4);
    uint4 i1 = uint4(5, 6, 7, 8);
    uint4 i2 = uint4(9, 10, 11, 12);
    uint4 magic = uint4(0xA5A5A5A5, 0x5A5A5A5A, 0xFF00FF00, 0x00FF00FF);

    [unroll({unroll_factor})]
    for (int i = 0; i < MAX_STEPS; i++) {{
        f0 = mad(f1, c1, f2);
        f1 = mad(f2, c1, f3);
        f2 = mad(f3, c1, f0);
        f3 = mad(f0, c1, f1);

        i0 = (i0 << 1) ^ i1;
        i1 = (i1 >> 1) ^ i2;
        i2 = (i2 << 3) ^ i0 ^ magic;

        if ((i & 127) == 0) {{
            f0 = sin(f0) * cos(f1);
            f3 += (float4)i0 * 0.00001;
        }}
    }}
    return f0 + f1 + f2 + f3 + (float4)i0 + (float4)i1;
}}
"""

# RAM-heavy template using large static array and pseudo-random indexing
SHADER_TEMPLATE_RAM = """
struct PS_INPUT {{ float4 Pos : SV_POSITION; float2 UV : TEXCOORD; }};
static const float4 DATA_BLOCK[{array_size}] = {{ {array_data} }};
float4 PSMain(PS_INPUT input) : SV_TARGET {{
    uint idx = (uint)(abs(input.Pos.x) * 100000.0) % {array_size};
    float4 r0 = float4(input.UV, 0.5, 0.5);
    float4 r1 = float4(input.UV.yx, 0.2, 0.8);
    float4 acc = float4(0,0,0,0);
    [loop]
    for (int i = 0; i < {steps}; i++) {{
        uint j = (idx + (uint)(i * 127)) % {array_size};
        float4 m = DATA_BLOCK[j];
        r0 = mad(r1, r0, m);
        acc += r0 * 0.0001;
    }}
    return acc;
}}
"""

# Extra template: more transcendentals + control flow, smaller iteration count
SHADER_TEMPLATE_TRANS = """
#define MAX_STEPS {steps}
struct PS_INPUT {{ float4 Pos : SV_POSITION; float2 UV : TEXCOORD; }};
float hash(float2 p) {{
    return frac(sin(dot(p, float2(12.9898,78.233))) * 43758.5453);
}}
float4 PSMain(PS_INPUT input) : SV_TARGET {{
    float2 uv = input.UV * 10.0;
    float4 acc = float4(0,0,0,0);
    float4 r = float4(uv, 1.0, 0.0);
    [loop]
    for (int i = 0; i < MAX_STEPS; ++i) {{
        float h = hash(uv + i);
        r.xy = float2(cos(h * 6.2831), sin(h * 6.2831));
        r.zw = float2(tan(h), atan2(r.x, r.y));
        acc += sin(r * 0.1) * cos(r.yxwz * 0.13);
        if ((i & 63) == 0) {{
            uv = uv.yx * 1.0001 + h;
        }}
    }}
    return acc * 0.001 + float4(0.1,0.2,0.3,0.4);
}}
"""

RAM_ARRAY_SIZE = 16384
_ram_data_str = None


def build_ram_data():
    global _ram_data_str
    if _ram_data_str is not None:
        return _ram_data_str
    items = []
    for _ in range(RAM_ARRAY_SIZE):
        items.append(
            "float4({:.6f},{:.6f},{:.6f},{:.6f})".format(
                random.random(),
                random.random(),
                random.random(),
                random.random(),
            )
        )
    _ram_data_str = ",\n".join(items)
    return _ram_data_str


# --- DXC HELPER ---

def download_and_setup_dxc():
    bin_path = os.path.abspath(os.path.join(DXC_LOCAL_DIR, "bin", "dxc.exe"))
    if os.path.exists(bin_path):
        return bin_path
    print(f"{Colors.WARNING}dxc.exe not found. Downloading...{Colors.ENDC}")
    try:
        if not os.path.exists(DXC_LOCAL_DIR):
            os.makedirs(DXC_LOCAL_DIR)
        req = urllib.request.Request(
            DXC_DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for file in z.namelist():
                if file.startswith("bin/x64/"):
                    z.extract(file, DXC_LOCAL_DIR)
        x64_dir = os.path.join(DXC_LOCAL_DIR, "bin", "x64")
        target_bin = os.path.join(DXC_LOCAL_DIR, "bin")
        if os.path.exists(x64_dir):
            for f in os.listdir(x64_dir):
                shutil.move(os.path.join(x64_dir, f), target_bin)
            os.rmdir(x64_dir)
        return bin_path
    except Exception as e:
        print(f"{Colors.FAIL}Download failed: {e}{Colors.ENDC}")
        sys.exit(1)


def get_dxc_path(user_arg):
    if user_arg and os.path.exists(user_arg):
        return user_arg
    local_dxc = os.path.abspath(os.path.join(DXC_LOCAL_DIR, "bin", "dxc.exe"))
    if os.path.exists(local_dxc):
        return local_dxc
    sys_dxc = shutil.which("dxc.exe")
    if sys_dxc:
        return sys_dxc
    return download_and_setup_dxc()


# --- SHADER GENERATION ---

def generate_shader(filename, seed):
    random.seed(seed)
    roll = random.random()
    # ~30% RAM-heavy, ~50% hybrid, ~20% extra transcendental/control-flow heavy
    if roll < 0.30:
        data = build_ram_data()
        steps = random.randint(60000, 100000)
        code = SHADER_TEMPLATE_RAM.format(
            array_size=RAM_ARRAY_SIZE,
            array_data=data,
            steps=steps,
        )
    elif roll < 0.80:
        steps = random.randint(600000, 1000000)
        unroll = 1024
        code = SHADER_TEMPLATE_HYBRID.format(
            steps=steps,
            unroll_factor=unroll,
        )
    else:
        steps = random.randint(150000, 300000)
        code = SHADER_TEMPLATE_TRANS.format(steps=steps)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)


def prepare_workload(count):
    sys.stdout.write(f"Pre-generating {count} shaders... ")
    sys.stdout.flush()
    files = []
    for i in range(count):
        fname = os.path.join(TEMP_DIR, f"stress_{i}.hlsl")
        generate_shader(fname, i)
        files.append(fname)
    print(f"{Colors.OKGREEN}Done.{Colors.ENDC}")
    return files


def compile_once(dxc_path, shader_file):
    try:
        flags = (
            0x00004000 | 0x00000008
            if sys.platform == "win32"
            else 0
        )
        cmd = [
            dxc_path,
            "-T",
            "ps_6_6",
            "-O3",
            "-Vd",
            "-E",
            "PSMain",
            "-HV",
            "2021",
            "-all_resources_bound",
            shader_file,
            "-Fo",
            "NUL",
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=flags,
        )
        if result.returncode != 0:
            if (
                result.returncode in CRASH_EXIT_CODES
                or abs(result.returncode) > 1000000
            ):
                return False, f"CRASH: {result.returncode}", result.returncode
            return False, "Error", result.returncode
        return True, None, 0
    except Exception as e:
        return False, str(e), -1


# --- NOISE / STRESS PROCESSES ---

class RamAnvil:
    def __init__(self, stop_event, noise_event):
        self.stop_event = stop_event
        self.noise_event = noise_event

    def start_loop(self):
        try:
            total = get_total_ram()
            size = int(total * 0.70)
            buf = bytearray(size)
            for i in range(0, size, 4096 * 100):
                buf[i] = 1
        except Exception:
            return

        rng = random.Random()
        view = memoryview(buf)
        max_idx = size - 4096

        page_size = 4096
        page_count = size // page_size
        if page_count > 0:
            idx_table = list(range(page_count))
            rng.shuffle(idx_table)
        else:
            idx_table = []
        cur_page = 0

        while not self.stop_event.is_set():
            if not self.noise_event.is_set():
                time.sleep(0.1)
                continue
            try:
                mode = rng.randint(0, 9)
                if mode < 5:
                    idx = rng.randint(0, max_idx)
                    _ = view[idx]
                    view[idx] = rng.randint(0, 255)
                elif mode < 8 and idx_table:
                    cur_page = idx_table[cur_page]
                    base = cur_page * page_size
                    off = rng.randint(0, page_size - 1)
                    view[base + off] ^= 0xFF
                else:
                    base = rng.randint(0, max_idx)
                    end = min(base + 4096, size)
                    for i in range(base, end, 128):
                        view[i] = (view[i] + 1) & 0xFF
            except Exception:
                break


class CacheTrasher:
    def __init__(self, stop_event, noise_event):
        self.stop_event = stop_event
        self.noise_event = noise_event
        self.raw = ctypes.create_string_buffer(128)
        self.align = (ctypes.addressof(self.raw) + 63) & ~63

    def _thrasher(self):
        while not self.stop_event.is_set():
            if not self.noise_event.is_set():
                time.sleep(0.1)
                continue
            ctypes.memset(self.align, 1, 1)

    def start_loop(self):
        t = threading.Thread(target=self._thrasher, daemon=True)
        t.start()
        t.join()


class IntegrityStress:
    def __init__(self, stop_event, q, noise_event):
        self.stop_event = stop_event
        self.q = q
        self.noise_event = noise_event

    def start_loop(self):
        raw = os.urandom(1024) * 16384  # 16 MiB
        crc_expected = zlib.crc32(raw) & 0xFFFFFFFF
        blake_expected = hashlib.blake2b(raw, digest_size=32).digest()

        comp_z = zlib.compress(raw, level=6)
        comp_lzma = lzma.compress(raw, preset=6)

        while not self.stop_event.is_set():
            if not self.noise_event.is_set():
                time.sleep(0.1)
                continue
            try:
                # zlib path
                dec_z = zlib.decompress(comp_z)
                crc_z = zlib.crc32(dec_z) & 0xFFFFFFFF
                b_z = hashlib.blake2b(dec_z, digest_size=32).digest()

                # LZMA path
                dec_l = lzma.decompress(comp_lzma)
                crc_l = zlib.crc32(dec_l) & 0xFFFFFFFF
                b_l = hashlib.blake2b(dec_l, digest_size=32).digest()

                if (
                    crc_z != crc_expected
                    or b_z != blake_expected
                    or crc_l != crc_expected
                    or b_l != blake_expected
                ):
                    self.q.put(
                        (
                            0xC0000428,
                            "INTEGRITY FAILURE",
                            "Checksum/Hash Mismatch",
                        )
                    )
                    self.stop_event.set()
                    break
            except Exception:
                pass


class IoStress:
    def __init__(self, stop_event, d, noise_event, s=1024):
        self.stop_event = stop_event
        self.f = os.path.join(d, "io.dat")
        self.s = s * 1024 * 1024  # MiB -> bytes
        self.noise_event = noise_event

    def start_loop(self):
        if not os.path.exists(self.f):
            with open(self.f, "wb") as f:
                f.write(os.urandom(self.s))

        k32 = ctypes.windll.kernel32
        HANDLE = wintypes.HANDLE

        k32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            HANDLE,
        ]
        k32.CreateFileW.restype = HANDLE

        k32.ReadFile.argtypes = [
            HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        k32.ReadFile.restype = wintypes.BOOL

        h = k32.CreateFileW(
            self.f,
            0x80000000,
            0,
            None,
            3,
            0x20000000,
            None,
        )
        if h == HANDLE(-1).value:
            return

        buf = ctypes.create_string_buffer(8192)
        addr = (ctypes.addressof(buf) + 4095) & ~4095
        read = wintypes.DWORD()
        rng = random.Random()
        max_s = (self.s - 4096) // 4096

        try:
            while not self.stop_event.is_set():
                if not self.noise_event.is_set():
                    time.sleep(0.1)
                    continue
                k32.SetFilePointer(
                    h,
                    rng.randint(0, max_s) * 4096,
                    None,
                    0,
                )
                k32.ReadFile(
                    h,
                    ctypes.c_void_p(addr),
                    4096,
                    ctypes.byref(read),
                    None,
                )
        finally:
            k32.CloseHandle(h)


def noise_entry(stop, q, d, noise_event, target_noise_count):
    """
    Launches noise threads up to target_noise_count.
    Priority order if we must prune:
    1. Integrity (Primary)
    2. RamAnvil
    3. Integrity (Secondary)
    4. CacheTrasher
    5. IoStress
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        # List of potential threads in order of IMPORTANCE/PRIORITY
        # We add them to this list, then slice based on count
        candidates = []

        # 1. Primary Integrity (Crucial)
        candidates.append(IntegrityStress(stop, q, noise_event))
        # 2. RAM Anvil (Crucial)
        candidates.append(RamAnvil(stop, noise_event))
        # 3. Secondary Integrity (Very Good)
        candidates.append(IntegrityStress(stop, q, noise_event))
        # 4. Cache Thrashing (Good)
        candidates.append(CacheTrasher(stop, noise_event))
        # 5. IO Stress (Lowest Priority if cores are starved)
        candidates.append(IoStress(stop, d, noise_event))

        # Select top N threads
        active_workers = candidates[:target_noise_count]

        threads = []
        for worker in active_workers:
            t = threading.Thread(target=worker.start_loop, daemon=True)
            threads.append(t)
            t.start()

        while not stop.is_set():
            time.sleep(0.5)
    except Exception:
        pass


# --- MAIN PROCESS STATE ---

stats_lock = threading.Lock()
compiled_count = 0
error_count = 0
active_compiles = 0
target_active = 0
crashed = False
crash_info = None
pulse_barrier = None


def worker_loop(dxc, pool, mode, stop):
    global compiled_count, error_count, active_compiles, crashed, crash_info
    rng = random.Random()
    next_pulse = time.time() + random.uniform(0, PULSE_INTERVAL)

    while not stop.is_set():
        if mode == "variable" and pulse_barrier and time.time() > next_pulse:
            try:
                pulse_barrier.wait(timeout=2.0)
            except Exception:
                pass
            time.sleep(PULSE_DURATION)
            next_pulse = (
                time.time() + PULSE_INTERVAL + random.uniform(-1, 1)
            )

        wait = False
        if mode == "variable":
            with stats_lock:
                if active_compiles >= target_active:
                    wait = True
                else:
                    active_compiles += 1
        else:
            with stats_lock:
                active_compiles += 1

        if crashed:
            break

        if wait:
            for _ in range(1000):
                pass
            continue

        success, msg, code = compile_once(dxc, rng.choice(pool))

        with stats_lock:
            active_compiles -= 1
            if not success:
                if code in CRASH_EXIT_CODES or (
                    code and abs(code) > 1000000
                ):
                    crashed = True
                    crash_info = (
                        code,
                        CRASH_EXIT_CODES.get(code, "Unknown"),
                        msg,
                    )
                    stop.set()
                elif code:
                    error_count += 1
            else:
                compiled_count += 1


def get_target(elapsed, total, mode):
    if mode in ["steady", "benchmark"]:
        return total, "STEADY MAX", True
    t = elapsed % CYCLE_DURATION

    if t < 20:
        return total, "HEAT SOAK", True
    if t < 35:
        return (0 if int(t) % 2 == 0 else total), "SPIKE WAVE", True
    if t < 50:
        random.seed(int(t * 10))
        choice = random.choice(
            [
                (0, "CHAOS 0%"),
                (max(1, total // 2), "CHAOS 50%"),
                (total, "CHAOS 100%"),
            ]
        )
        return choice[0], choice[1], True

    tgt = 1 if int(t) % 2 == 0 else 2
    return tgt, f"BOOST ({tgt}T)", False


def real_main(input_args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dxc")
    p.add_argument("--threads", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--mode", default="steady")
    p.add_argument("--duration", type=int, default=0)
    p.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    # Internal arg to pass noise count to subprocess
    p.add_argument("--noisecount", type=int, default=MAX_NOISE_THREADS, help=argparse.SUPPRESS)

    if input_args:
        args = p.parse_args(input_args)
    else:
        args = p.parse_args()

    if args.worker and args.noisecount:
        pass

    # --- THREAD & NOISE CALCULATION ---
    # 1. Determine how many Noise threads we CAN run
    # Rule: Keep at least 1 compiler thread.

    actual_noise_threads = 0

    if args.mode != "benchmark":
        # If we have 1 core, noise=0. If 6+, noise=5.
        actual_noise_threads = min(
            MAX_NOISE_THREADS, max(0, LOGICAL_CORES - 1)
        )

    # 2. Determine Compiler Threads
    actual_threads = args.threads
    if actual_threads == -1:
        if args.mode == "benchmark":
            actual_threads = LOGICAL_CORES
        else:
            # User requested Cores - 3.
            # This generally results in Cores + 2 total threads if Noise=5
            actual_threads = max(1, LOGICAL_CORES - 3)

    global target_active, pulse_barrier
    if args.mode == "variable":
        pulse_barrier = threading.Barrier(actual_threads)

    dxc = get_dxc_path(args.dxc)
    print(f"{Colors.HEADER}=== UE5 Stress: FINAL EDITION (Universal V9) ==={Colors.ENDC}")
    print(
        f"Threads: {actual_threads} (Compilers) + {actual_noise_threads} (Noise) "
        f"| Mode: {args.mode} | Priority: BELOW_NORMAL"
    )

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    pool = prepare_workload(max(200, actual_threads * 4))

    mp_stop = multiprocessing.Event()
    mp_q = multiprocessing.Queue()
    mp_noise_event = multiprocessing.Event()
    mp_noise_event.set()

    noise = None
    if actual_noise_threads > 0 and args.mode != "benchmark":
        noise = multiprocessing.Process(
            target=noise_entry,
            args=(mp_stop, mp_q, TEMP_DIR, mp_noise_event, actual_noise_threads),
        )
        noise.daemon = True
        noise.start()

    stop = threading.Event()
    workers = []
    with stats_lock:
        target_active = (
            actual_threads if args.mode != "variable" else 0
        )
    for _ in range(actual_threads):
        t = threading.Thread(
            target=worker_loop,
            args=(dxc, pool, args.mode, stop),
            daemon=True,
        )
        workers.append(t)
        t.start()

    start = time.time()
    rate_history = collections.deque()
    rate_history_60 = collections.deque()

    bench_start_cc = 0
    bench_block_start = start
    bench_scores = []
    best_score = 0.0

    last_rate_update = 0.0
    last_60s_update = 0.0
    display_rate = 0.0
    display_60s = "N/A"

    last_var_cc = 0
    last_var_time = start

    try:
        while not stop.is_set():
            now = time.time()
            elapsed = now - start
            if args.duration > 0 and elapsed > args.duration:
                stop.set()

            if not mp_q.empty():
                c = mp_q.get()
                global crashed, crash_info
                crashed = True
                crash_info = c
                stop.set()

            tgt, phase, noise_state = get_target(
                elapsed, actual_threads, args.mode
            )

            if args.mode != "benchmark" and noise:
                if noise_state:
                    mp_noise_event.set()
                else:
                    mp_noise_event.clear()

            if args.mode == "variable":
                with stats_lock:
                    target_active = tgt
            else:
                phase = "STEADY MAX"

            with stats_lock:
                ac = active_compiles
                cc = compiled_count
                ec = error_count

            calculated_rate = 0.0

            if args.mode in ["steady", "benchmark"]:
                if args.mode == "benchmark":
                    if (now - bench_block_start) >= 60.0:
                        block_rate = (cc - bench_start_cc) / (
                            now - bench_block_start
                        )
                        bench_scores.append(block_rate)
                        best_score = max(bench_scores)
                        bench_block_start = now
                        bench_start_cc = cc

                if (
                    not rate_history
                    or (now - rate_history[-1][0]) > 0.25
                ):
                    rate_history.append((now, cc))
                    while (
                        rate_history
                        and rate_history[0][0] < now - 30.0
                    ):
                        rate_history.popleft()
                    rate_history_60.append((now, cc))
                    while (
                        rate_history_60
                        and rate_history_60[0][0] < now - 60.0
                    ):
                        rate_history_60.popleft()

                if len(rate_history) > 1:
                    d_t = rate_history[-1][0] - rate_history[0][0]
                    d_c = rate_history[-1][1] - rate_history[0][1]
                    calculated_rate = d_c / d_t if d_t > 0 else 0.0

                if elapsed > 60.0 and (
                    now - last_60s_update
                ) > 60.0:
                    if len(rate_history_60) > 1:
                        d60_t = (
                            rate_history_60[-1][0]
                            - rate_history_60[0][0]
                        )
                        d60_c = (
                            rate_history_60[-1][1]
                            - rate_history_60[0][1]
                        )
                        val = (
                            d60_c / d60_t if d60_t > 0 else 0.0
                        )
                        display_60s = f"{val:.1f}/s"
                    last_60s_update = now

                update_interval = 2.0
            else:
                update_interval = 0.5

            if now - last_rate_update > update_interval:
                if args.mode == "variable":
                    dt = now - last_var_time
                    dc = cc - last_var_cc
                    display_rate = dc / dt if dt > 0 else 0.0
                    last_var_time = now
                    last_var_cc = cc
                else:
                    display_rate = calculated_rate
                last_rate_update = now

            io_s = ""
            int_s = ""
            active_noise_count_display = 0

            if args.mode != "benchmark":
                is_active = mp_noise_event.is_set()
                if is_active:
                    active_noise_count_display = actual_noise_threads

                io_s = (
                    f" {Colors.FAIL}[IO: ACT]{Colors.ENDC}"
                    if is_active
                    else " [IO: OFF]"
                )
                int_s = (
                    f" {Colors.OKBLUE}[INT: OK]{Colors.ENDC}"
                    if is_active
                    else " [INT: PAUSE]"
                )

            total_active_display = ac + active_noise_count_display

            extra_info = ""
            if args.mode == "steady":
                extra_info = f" | 60s Avg: {display_60s}"
            elif args.mode == "benchmark":
                count_done = len(bench_scores)
                if count_done < 3:
                    extra_info = (
                        f" | Best 60s: {best_score:.1f}/s "
                        f"(Round {count_done+1}/3)"
                    )
                else:
                    extra_info = (
                        " | Best 60s: "
                        f"{Colors.OKGREEN}{best_score:.1f}/s"
                        f"{Colors.ENDC}"
                    )

            color = (
                Colors.OKGREEN
                if "IDLE" in phase
                else (
                    Colors.FAIL
                    if "SPIKE" in phase
                    else Colors.WARNING
                )
            )

            try:
                cols = shutil.get_terminal_size().columns
            except Exception:
                cols = 120

            stat = (
                f"Time: "
                f"{str(datetime.timedelta(seconds=int(elapsed)))} "
                f"| Rate: {display_rate:.1f}/s"
                f"{extra_info} | Act: {total_active_display} | Err: {ec} | "
                f"{color}{phase}{Colors.ENDC}{io_s}{int_s}"
            )
            print(f"\r{stat:<{cols-1}}", end="", flush=True)

            if crashed:
                print(
                    f"\n\n{Colors.FAIL}{'='*60}\n"
                    f"FAILURE DETECTED\n{'='*60}{Colors.ENDC}"
                )
                if crash_info:
                    print(
                        f"Code: {crash_info[0]}\n"
                        f"Reason: {crash_info[1]}\n"
                        f"Msg: {crash_info[2]}"
                    )
                sys.exit(1)

            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        mp_stop.set()
        for t in workers:
            t.join(1.0)

        if noise and noise.is_alive():
            noise.terminate()
            noise.join(timeout=2.0)
            if noise.is_alive():
                try:
                    noise.kill()
                except AttributeError:
                    pass

        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("\nCleanup Done.")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception:
            pass


# --- WATCHDOG / FRONTEND ---

def run_watchdog():
    pass_args = sys.argv[1:]
    interactive_mode = False

    if len(sys.argv) == 1:
        interactive_mode = True
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print(
                    f"{Colors.HEADER}========================================================"
                )
                print("                       PYBECRASHER                      ")
                print(
                    "    UE5 SHADER COMPILATION AND OODLE-LIKE STRESSOR      "
                )
                print(
                    f"========================================================"
                    f"{Colors.ENDC}"
                )
                print("")
                print(
                    "This tool simulates the Unreal Engine 5 "
                    '"Preparing Shaders"'
                )
                print(
                    "workload plus heavy asset decompression and RAM noise"
                )
                print("to test CPU/RAM/IMC stability.")
                print("")
                print("Select Mode:")
                print("")
                print(f"{Colors.OKBLUE}[1]{Colors.ENDC} Variable \"Chaos\" Load")
                print("    - Cycles between Single Core, Ramp Up, Random, and")
                print(
                    "      Transient Spikes (Idle to Max instantly) with "
                    "noise."
                )
                print("")
                print(f"{Colors.OKBLUE}[2]{Colors.ENDC} Steady Load")
                print("    - Constant 100% compiler load + noise.")
                print("    - Best for thermal and long-term stability.")
                print("")
                print(f"{Colors.OKBLUE}[3]{Colors.ENDC} Benchmark Mode")
                print("    - Pure Compiler Throughput (No Noise).")
                print("    - Best for comparing CPU performance.")
                print("")
                print("--- Info ---")
                print(f"{Colors.OKBLUE}[4]{Colors.ENDC} View README.md")
                print(f"{Colors.OKBLUE}[5]{Colors.ENDC} View LICENSE")
                print("")

                choice = input("Enter selection (default is 1): ").strip()

                if choice == "4":
                    print_file_content("README.md", "README")
                    continue
                elif choice == "5":
                    print_file_content("LICENSE", "LICENSE")
                    continue
                elif choice == "2":
                    pass_args = ["--mode", "steady"]
                    break
                elif choice == "3":
                    pass_args = ["--mode", "benchmark"]
                    break
                else:
                    pass_args = ["--mode", "variable"]
                    break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return

        print("")
        print(
            f"Starting Stress Test in {pass_args[1]} mode..."
        )
        print("Press Ctrl+C to stop at any time.")
        print("")

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--worker"] + pass_args
    else:
        script = os.path.abspath(sys.argv[0])
        cmd = [sys.executable, script, "--worker"] + pass_args

    aborted = False
    proc = None
    try:
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0
        )
        proc = subprocess.Popen(cmd, creationflags=creationflags)

        while proc.poll() is None:
            time.sleep(0.5)

        if proc.returncode not in (0, 3221225786):
            print(
                f"\n{Colors.FAIL}CRITICAL FAILURE: Exit Code "
                f"{proc.returncode}{Colors.ENDC}"
            )
            if proc.returncode in CRASH_EXIT_CODES:
                print(
                    f"Reason: {CRASH_EXIT_CODES[proc.returncode]}"
                )
    except KeyboardInterrupt:
        aborted = True
        if proc is not None:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()

            print("\nStopping...")
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    if interactive_mode and not aborted:
        print("")
        print("========================================================")
        print("Test execution finished.")
        print("========================================================")
        input("Press Enter to close...")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--worker" in sys.argv:
        real_main(sys.argv[1:])
    else:
        run_watchdog()
