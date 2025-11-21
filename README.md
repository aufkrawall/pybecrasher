# pybecrasher
Vibe-scripted Python-based CPU stress test mimicking UE5 workloads
<img width="867" height="464" alt="pybecrasher" src="https://github.com/user-attachments/assets/c91250ce-a98b-4388-a02c-98e5ecb9dd8a" />

Provided as-is, I DO NOT ACCEPT ANY LIABILITY for any issues and damages caused by this tool. 

AI-generated description:
## ⚙️ Script Functionality and Process

The primary function of this script is to execute a rigorous, controlled hardware stress test by simulating the most resource-intensive aspects of a game engine's asset pipeline.

***

### **Core Execution Flow**

1.  **Dependency Setup:**
    * The script checks for the presence of the **DirectX Shader Compiler (`dxc.exe`)**.
    * If absent, it automatically **downloads the official Microsoft binary** from GitHub and prepares it for use.

2.  **Architecture:**
    * It utilizes a **multiprocessing architecture** to run the heavy **Noise Process** (I/O, RAM, Cache) completely separate from the **Main Worker Process** (compilation threads). This bypasses Python's Global Interpreter Lock (GIL) and guarantees high CPU saturation.

3.  **Workload Generation:**
    * It creates a pool of synthetic HLSL files in a temporary directory (`temp_shaders`).
    * Shaders are optimized for maximum transistor usage, falling into three categories: **Hybrid FMA** (heavy floating-point + integer math), **LZ Mimic** (bitwise/integer density), or **RAM Eater** (memory access contention).

***

## 🔬 Technical Mechanisms

| Feature | Action | Stress Target |
| :--- | :--- | :--- |
| **Utilization & Power** | **Fixed-Point Workload** (Shader Complexity) is increased to minimize kernel overhead, maximize instruction throughput (IPC), and ensure maximum CPU utilization. | CPU Core Utilization & Thermal Limits |
| **RAM Anvil** | Allocates **70% of physical RAM** and performs continuous random writes. | Memory Controller (IMC) & DRAM Thermals. |
| **Cache Thrashing** | Forces writes to the same 64-byte shared memory address (False Sharing). | CPU Ring Bus / Inter-core Coherency. |
| **Power Pulse** | In `variable` mode, all compiler threads are synchronized to instantly drop load and resume. | **VRM Transient Response** (Voltage Droop). |
| **Direct I/O** | Uses Win32 API (`FILE_FLAG_NO_BUFFERING`) for unbuffered random reads. | NVMe Controller and PCIe Bus stability. |
| **Integrity Check** | Runs zlib decompression loops and verifies checksums. | **ALU Stability** (detects single-bit calculation errors). |

## 🚀 USAGE AND EXECUTION

### Prerequisites
* Windows 10/11 (64-bit).
* Windows page file must be present to avoid oom crashes, pybecrasher causes high virtual memory allocation. Maybe even increase minimum page file size.
* Make sure your cooling and power supply are sufficient for heavy stresstesting.
* Python 3.6+ installed and available in the system PATH (normal installation, straightforward procedure).
* My Zen 3 CPU just instantly reboots the entire system when I test it with too low voltage, so I couldn't test error detection feature too extensively.

### Running the Test

1.  Place the included `run.bat` and `ue5_stress.py` files in a local folder.
2.  Open `run.bat`, no admin privileges required.


It might actually be really useful, or total garbage. I do not provide updates, nor fix bugs, except when I do.
