# pybecrasher
Vibe-scripted Python-based CPU stress test mimicking UE5 workloads
<img width="867" height="464" alt="pybecrasher" src="https://github.com/user-attachments/assets/c91250ce-a98b-4388-a02c-98e5ecb9dd8a" />

> **⚠️ DISCLAIMER:** Provided as-is. I DO NOT ACCEPT ANY LIABILITY for issues, hardware damage, OS corruption, or data loss caused by this tool.

AI-generated and edited description:
## Script Functionality and Process

The primary function of this script is to execute a rigorous, controlled hardware stress test by simulating the most resource-intensive aspects of a game engine's asset pipeline.

## ⚙️ Internal Mechanics & Architecture

### 1. Compiler & Dependency Management
* **Automatic Bootstrapping:** The script does not ship with binaries. On the first run, it detects if `dxc.exe` is missing.
* **Retrieval:** It downloads the official **Microsoft DirectX Shader Compiler (DXC)** release (v1.8.2407) directly from GitHub, extracts the specific x64 binary to a local `dxc_bin` folder, and configures the environment path dynamically.
* **Usage:** It launches `dxc.exe` as a subprocess for every single compilation task, forcing the OS kernel to handle rapid process creation/destruction, which adds significant overhead stress often missing from synthetic benchmarks.

### 2. Synthetic Workload Generation (The "Fake" UE5)
Instead of compiling actual game assets, the script generates **synthetic HLSL shaders**:
* **Shaders:** These shaders mix **Floating Point (AVX/FMA)** and **Integer (Bitwise/ALU)** instructions in parallel.
* **Complexity:** Shaders utilize massive loop counts (up to 1,000,000 steps) and `#unroll` directives to force the compiler to generate huge instruction streams, maximizing the time the CPU spends in "User Mode" doing math.
* **Randomization:** A pool of unique shaders is pre-generated at startup. Worker threads randomly select a file from this pool for every job. This (hopefully) thrashes the CPU branch predictor and prevents the OS from caching the compilation result effectively.

### 3. Architecture: Bypassing the Python Bottleneck
Python's Global Interpreter Lock (GIL) usually prevents 100% CPU utilization in multithreaded apps. This script circumvents this:
* **Process Isolation:** It spawns a completely separate **Noise Process** (`multiprocessing`) to handle I/O, RAM, and Integrity checks. This ensures the **Main Worker Process** has zero contention and can dedicate 100% of its GIL time to spawning compiler threads.
* **Busy Wait Strategy:** In "Variable" mode, threads utilize busy-wait loops (burning cycles) instead of sleeping, ensuring the CPU core never enters a low-power C-state even while waiting for a task slot.

### 4. Subsystem Stress Modules
* **Oodle Simulation (ALU Integrity):** The script uses `zlib` decompression in a tight loop as a proxy for Oodle/Kraken workloads. It verifies the CRC32 checksum of every decompressed block. On errors, this thread will crash with an **Integrity Failure**.
* **I/O Stress (PCIe/NVMe):** Every 30 seconds, a background thread wakes up and hammers the drive (read only, creates a 1GB file for this once). It uses `ctypes` to call the Win32 API `CreateFileW` with `FILE_FLAG_NO_BUFFERING`. This aims to bypass the Windows RAM cache, hopefully forcing physical 4KB random reads from the SSD controller, generating heat and PCIe bus traffic.
* **RAM Anvil:** Allocates ~70% of total physical RAM and periodically writes to random pages. This tests the Memory Controller (IMC) under load and checks for thermal instability in DIMMs.

### Interpreting Crashes
The script employs a **Watchdog** wrapper. If the test closes, check the output:
* `0xC0000005` (Access Violation): Usually VCore too low or RAM unstable.
* `0xC0000428` (Integrity Checksum): CPU/Cache calculation error (Core instability).
* **Hard Reboot:** PSU or severe voltage instability (Vdroop).

### Modes
* **Variable Mode:** Introduces a "Power Virus Pulse" where all threads synchronize to drop load and spike simultaneously every 5 seconds. This tests **Transient Response** (VRM voltage regulation stability).
* **Steady Mode:** Attempts to pin the CPU to 100% usage constantly. The "Rate" displayed is a **rolling average over the last 5 seconds**, giving you real-time feedback on compile speed as seen in games.

## 🚀 USAGE AND EXECUTION

### Prerequisites
* Windows 10/11 (64-bit).
* Windows page file must be present to avoid oom crashes, pybecrasher causes high virtual memory allocation. Maybe even increase minimum page file size.
* Make sure your cooling and power supply are sufficient for heavy stresstesting.
* Python 3.6+ installed and available in the system PATH (normal installation, straightforward procedure).
* My Zen 3 CPU just instantly reboots the entire system when I test it with too low voltage, so I couldn't test error detection feature too extensively.
* Try 2 hours of stress testing. The variable load mode probably is more meaningful than steady load mode. Though steady load mode might be a neat benchmark for shader compile performance.

### Running the Test

1.  Place the included `run.bat` and `ue5_stress.py` files in a local folder.
2.  Open `run.bat`, no admin privileges required.


It might actually be really useful, or total garbage. I do not provide updates, nor fix bugs, except when I do.
