# pybecrasher
Vibe-scripted Python-based CPU stress test mimicking UE5 workloads
<img width="867" height="464" alt="pybecrasher" src="https://github.com/user-attachments/assets/c91250ce-a98b-4388-a02c-98e5ecb9dd8a" />

> **⚠️ DISCLAIMER:** Provided as-is. I DO NOT ACCEPT ANY LIABILITY for issues, hardware damage, OS corruption, or data loss caused by this tool.

AI-generated and edited description:
## Script Functionality and Process

The primary function of this script is to execute a controlled hardware stress test by simulating the most resource-intensive aspects of a game engine's asset pipeline.

## 🛡️ Safety & Security

### 1. Antivirus False Positives
If you download the binary version (`.exe`), your antivirus (e.g., Windows Defender) may flag it as generic malware (often `Trojan:Win32/Wacatac`).
*   **Why?** This tool is packaged using **PyInstaller**. Because malware authors often use PyInstaller to hide their code, antivirus heuristics mistakenly flag *any* unsigned PyInstaller executable as suspicious.
*   **Verification:** You can verify the safety of this tool by ignoring the `.exe` and running the raw Python script (`ue5_shader_stress.py`) directly. The source code is open, transparent, and readable.

### 2. External Downloads (DXC.exe)
This tool requires the **Microsoft DirectX Shader Compiler**. It does **not** ship with this binary included.
*   **Process:** Upon first run, the script automatically downloads `dxc.exe`.
*   **Source:** The download occurs **directly** from Microsoft's official GitHub repository (`github.com/microsoft/DirectXShaderCompiler`).
*   **Safety:** The script does not connect to any third-party mirrors or private servers.

---

## ⚙️ Internal Mechanics & Architecture

### 1. Compiler & Dependency Management
* **Automatic Bootstrapping:** The script does not ship with binaries. On the first run, it detects if `dxc.exe` is missing.
* **Retrieval:** It downloads the official **Microsoft DirectX Shader Compiler (DXC)** release (v1.8.2407) directly from GitHub, extracts the specific x64 binary to a local `dxc_bin` folder, and configures the environment path dynamically.
* **Usage:** It launches `dxc.exe` as a subprocess for every single compilation task, forcing the OS kernel to handle rapid process creation/destruction, which adds significant overhead stress often missing from synthetic benchmarks.

### 2. Synthetic Workload Generation (The "Fake" UE5)
Instead of compiling actual game assets, the script generates **synthetic HLSL shaders**:
* **Shaders:** These shaders mix **Floating Point (AVX/FMA)** and **Integer (Bitwise/ALU)** instructions in parallel.
* **Complexity:** Shaders utilize massive loop counts (up to 1,000,000 steps) and `#unroll` directives to force the compiler to generate huge instruction streams, maximizing the time the CPU spends in "User Mode" doing math.
* **Randomization:** Shaders for variable and steady modes are constantly randomy generated and unique. This (hopefully) thrashes the CPU branch predictor and prevents the OS from caching the compilation result effectively. Causes a few hundred KB/s storage writes in variable and steady mode.
* **Why it is only a partial real-world shader compile load:** This test employs frontend compiling (HLSL to DXIL/IR), whereas the GPU driver performs the backend compiling (DXIL to GPU ISA). This is as close as we can get to real-world loads, and hopefully should work well enough.

### 3. Architecture: Bypassing the Python Bottleneck
Python's Global Interpreter Lock (GIL) usually prevents 100% CPU utilization in multithreaded apps. This script circumvents this:
* **Process Isolation:** It spawns a completely separate **Noise Process** (`multiprocessing`) to handle I/O, RAM, and Integrity checks. This ensures the **Main Worker Process** has zero contention and can dedicate 100% of its GIL time to spawning compiler threads.
* **Busy Wait Strategy:** In "Variable" mode, threads utilize busy-wait loops (burning cycles) instead of sleeping, ensuring the CPU core never enters a low-power C-state even while waiting for a task slot.

### 4. Subsystem Stress Modules
* **Oodle Simulation (ALU Integrity):** Runs parallel **zlib** and **LZMA** decompression loops, mimicking game asset loading.
* **I/O Stress (PCIe/NVMe):** A background thread constantly causes drive read-accesses (creates a 1GB file for this once). It uses `ctypes` to call the Win32 API `CreateFileW` with `FILE_FLAG_NO_BUFFERING`. This aims to bypass the Windows RAM cache, hopefully forcing physical 4KB random reads from the SSD controller, generating heat and PCIe bus traffic.
* **Integrity Checking:** Verifies data using **CRC32** and **BLAKE2b** hashes. A single flipped bit causes an immediate crash report.
* **RAM Anvil:** Allocates ~70% of total RAM. It performs **Random Pointer Chasing**, forcing the CPU to stall and wait for main memory, stressing the Memory Controller (IMC).

### Modes
* **Variable Mode:** Introduces a "Power Virus Pulse" where all threads synchronize to drop load and spike simultaneously every 5 seconds. This tests **Transient Response** (VRM voltage regulation stability).
* **Steady Mode:** Attempts to pin the CPU to 100% usage constantly.
* **Benchmark:** Pure compiler throughput test (Noise disabled). Uses deterministic seeding for consistent scores between runs.

### Logging
*   The script creates `pybecrasher.log` upon start.
*   If the test finishes successfully or is stopped by the user, **the log is automatically deleted**.
*   If a crash or integrity error occurs, the log is **saved** containing the specific error code and context. When things crash too hard/quickly, log files might not contain the actual crash code.

## 🚀 USAGE AND EXECUTION

### Prerequisites
* Windows 10/11 (64-bit).
* Windows page file must be present to avoid oom crashes, pybecrasher causes high virtual memory allocation. Maybe even increase minimum page file size.
* Make sure your cooling and power supply are sufficient for heavy stresstesting.
* Python 3.6+ installed and available in the system PATH (normal installation, straightforward procedure).
* My Zen 3 CPU just instantly reboots the entire system when I test it with too low voltage, so I couldn't test error detection feature too extensively.
* Try 2 hours of stress testing. The variable load mode probably is more meaningful than steady load mode.
* Exclude path from Windows Indexer and probably also avoid places like Desktop, where changes inside folders can cause weird refresh behavior.

### Running the Test

1.  Place the included `run.bat` and `ue5_stress.py` files in a local folder.
2.  Open `run.bat`, no admin privileges required.
