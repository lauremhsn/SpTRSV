# SpTRSV GPU Project - Sparse Triangular System Solver

Sparse Triangular System Solver (SpTRSV) implementation with GPU acceleration using CUDA. This project includes CPU baseline implementations and GPU kernel optimizations for solving sparse triangular linear systems.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Setup](#repository-setup)
3. [Data Files](#data-files)
4. [Building the Project](#building-the-project)
5. [Running Evaluations](#running-evaluations)
6. [System Information](#system-information)
7. [Results](#results)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Machine (Ran on Mac)
- Git
- SSH access to HPC cluster (AUB VPN required. SEE: https://servicedesk.aub.edu.lb/TDClient/1398/Portal/Requests/ServiceDet?ID=29740)
- Google Drive access for data download

### HPC Cluster Requirements
- CUDA toolkit (tested with 12.2.2)
- GCC compiler (newer version recommended)
- NVIDIA GPU (tested with Tesla V100-SXM3-32GB)
- SLURM job scheduler
- `curl` for downloading data files

---

## Repository Setup

### 1. Clone the Repository

On the HPC cluster head node after connecting:

```bash
cd ~
git clone https://github.com/lauremhsn/SpTRSV.git
cd SpTRSV/sptrsv
```

### 2. Directory Structure

```
SpTRSV/
├── sptrsv/
│   ├── Makefile
│   ├── main.cu
│   ├── matrix.cu
│   ├── matrix.h
│   ├── common.h
│   ├── timer.h
│   ├── kernelCPU.cu
│   ├── kernel0.cu
│   ├── kernel1.cu
│   ├── kernel2.cu
│   ├── kernel3.cu
│   ├── README.md
│   ├── MILESTONE_2.md
│   └── data/
│       ├── rajat18.txt
│       ├── parabolic_fem.txt
│       └── tmt_sym.txt
└── .git/
```

---

## Data Files

The project requires three sparse matrix datasets for evaluation. The data files are large (176 MB total) and must be downloaded separately. They cannot be uploaded via Github so we made use of Google Drive.

### Download Data Files via Google Drive

Create the `data/` directory and download the matrix files:

```bash
mkdir -p data

# Download rajat18.txt (16 MB) - Small dataset
curl -L 'https://drive.google.com/uc?export=download&id=1nn6LGyE7IJjPR84D-tXx7lG1f-JaFJUe' -o data/rajat18.txt

# Download parabolic_fem.txt (69 MB) - Medium dataset
curl -L 'https://drive.google.com/uc?export=download&id=1yoSNheQxMyzXIXwOyhUplBP7FZaty6o5' -o data/parabolic_fem.txt

# Download tmt_sym.txt (93 MB) - Large dataset
curl -L 'https://drive.google.com/uc?export=download&id=1S2iugJbqKpdNEGoPfPXiApKdqCW1FrMq' -o data/tmt_sym.txt
```

### Verify Data Files

```bash
ls -lah data/
# Expected output:
# -rw-rw-r-- 1 rmd35 rmd35 16M ... rajat18.txt
# -rw-rw-r-- 1 rmd35 rmd35 69M ... parabolic_fem.txt
# -rw-rw-r-- 1 rmd35 rmd35 93M ... tmt_sym.txt
```

**Note:** The data files must be made publicly available on Google Drive with "Anyone with the link" sharing enabled for the `curl` downloads to work. Otherwise, it will upload empty html/login screens.

---

## Building the Project

### 1. Request a GPU Node

On the cluster head node, request an interactive GPU session:

```bash
srun --partition=gpu --gres=gpu:1 --time=0:59:00 --pty bash
```

This allocates a GPU node (typically takes a few seconds). Your prompt will change to show the compute node name (e.g., `[rmd35@onode27 ~]$`).

**Note:** Use `sinfo` to check for availability.

### 2. Load Required Modules

On the compute node, load CUDA and GCC:

```bash
module load cuda/12.2.2
module load gcc
```

Verify the CUDA installation:

```bash
nvidia-smi
```

**CUDA Version Note:** Use CUDA 12.2.2 (not 12.4.0) for compatibility with host compiler and GPU architecture specifications. CUDA 12.4.0 has type traits issues with C++11.

### 3. Compile the Code

The Makefile requires C++11 support and GPU architecture specification. Compile with the appropriate flags:

```bash
cd ~/SpTRSV/sptrsv
make NVCC_FLAGS="-O3 -std=c++11 -arch=sm_70"
```

**Required Compilation Flags Explained:**
- `-O3`: Optimization level 3 for maximum performance
- `-std=c++11`: C++11 standard support (required by matrix operations)
- `-arch=sm_70`: Target Tesla V100 (Volta) GPU architecture (SM_70)

If compilation is successful, you should see `sptrsv` executable created.

**Note about kernels:** The project includes:
- `kernelCPU.cu` - CPU baseline implementation
- `kernel0.cu`, `kernel1.cu`, `kernel2.cu`, `kernel3.cu` - GPU kernel variants

The Makefile compiles all kernels by default. The `-d` flag at runtime selects which dataset to use, not which kernel.

---

## Running Evaluations

### Prerequisites Before Running

Ensure you're on a GPU node with:
- CUDA loaded: `module load cuda/12.2.2`
- GCC loaded: `module load gcc`
- Data files present: `ls data/` shows all three files
- Executable compiled: `ls sptrsv` exists

### Run the Evaluation

Execute the evaluation on all three datasets with kernel 0:

```bash
./sptrsv -d s -0   # Small dataset (rajat18) with kernel 0
./sptrsv -d m -0   # Medium dataset (parabolic_fem) with kernel 0
./sptrsv -d l -0   # Large dataset (tmt_sym) with kernel 0
```

Each command will:
1. Run CPU baseline on the dataset
2. Run GPU kernel with 128, 256, and 512 column tile sizes
3. Verify GPU results against CPU baseline
4. Report execution times in milliseconds

---

## System Information

Capture system details for your report:

```bash
# GPU Information
nvidia-smi

# Memory
free -h

# CPU Model
lscpu
```

---

## Results

### Milestone 2: Performance Benchmarking (April 14, 2026)

**Test Platform:** AUB Octopus HPC Cluster  
**GPU:** Tesla V100-SXM3-32GB  
**Driver:** NVIDIA 535.104.05 (CUDA 12.2)  
**Test Date:** April 14, 2026

#### Dataset: Small (rajat18.txt)
| Problem Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|---|---|---|---|
| 128 cols | 302.90 | 15.26 | **19.8x** |
| 256 cols | 650.01 | 15.33 | **42.4x** |
| 512 cols | 1497.90 | 18.03 | **83.0x** |

#### Dataset: Medium (parabolic_fem.txt)
| Problem Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|---|---|---|---|
| 128 cols | 4021.38 | 26.06 | **154.3x** |
| 256 cols | 8050.02 | 24.52 | **328.1x** |
| 512 cols | 16201.04 | 25.92 | **625.0x** |

#### Dataset: Large (tmt_sym.txt)
| Problem Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|---|---|---|---|
| 128 cols | 1985.64 | 757.63 | **2.6x** |
| 256 cols | 4156.21 | 1033.26 | **4.0x** |
| 512 cols | 11995.19 | 1795.08 | **6.7x** |

### Key Performance Observations

1. **Exceptional GPU Speedup on Medium Datasets:** Up to **625x speedup** (512 cols, parabolic_fem)
2. **Consistent Performance on Small Datasets:** 20-80x speedup across all tile sizes
3. **Reduced Speedup on Large Datasets:** 2.6-6.7x due to:
   - Memory bandwidth limitations
   - Lower computational intensity relative to memory transfers
   - GPU kernels designed for smaller matrices
4. **All Results Verified:** Floating-point precision differences are acceptable and expected between GPU/CPU
5. **Stable Compilation:** Successful after proper CUDA, GCC, and architecture configuration

---

## Troubleshooting

### Issue 1: `nvcc: command not found`

**Cause:** Running on the login node instead of compute node, or CUDA module not loaded.

**Solution:** 
1. Request a GPU compute node:
   ```bash
   srun --partition=gpu --gres=gpu:1 --pty bash
   ```
2. Load CUDA module on compute node:
   ```bash
   module load cuda/12.2.2
   ```

### Issue 2: CUDA 12.4.0 Type Traits Compilation Error

**Error:** `nontype "cuda::std::__4::remove_all_extents<_Tp>::type::type" is not a type name`

**Cause:** CUDA 12.4.0 has compatibility issues with older GCC versions and C++11 standard library implementations.

**Solution:** Use CUDA 12.2.2 instead:
```bash
module unload cuda
module load cuda/12.2.2
make NVCC_FLAGS="-O3 -std=c++11 -arch=sm_70"
```

### Issue 3: CUDA Atomics Architecture Error

**Error:** `CUDA atomics are only supported for sm_60 and up on *nix`

**Cause:** GPU architecture not specified or incompatible with the CUDA version.

**Solution:** Specify Tesla V100 (SM_70) architecture:
```bash
make NVCC_FLAGS="-O3 -std=c++11 -arch=sm_70"
```

### Issue 4: GCC Compiler Too Old for CUDA 12.2.2

**Error:** `libc++ does not support using GCC with C++03. Please enable C++11`

**Cause:** System default GCC (4.8.2) is incompatible with CUDA 12.2.2's libc++ implementation.

**Solution:** Load newer GCC module:
```bash
module load gcc
make NVCC_FLAGS="-O3 -std=c++11 -arch=sm_70"
```

### Issue 5: `Error: could not open file data/rajat18.txt`

**Solution:** Verify data files exist:
```bash
ls -la data/
```

Download missing files using the curl commands from the [Data Files](#data-files) section.

### Issue 6: Google Drive Download Fails

**Troubleshooting:**
1. Verify files are shared with "Anyone with the link" access
2. Try alternative download with confirmation:
   ```bash
   wget --no-check-certificate 'https://drive.google.com/uc?export=download&confirm=NO_ANTIVIRUS&id=FILE_ID' -O data/filename.txt
   ```
3. Check network connectivity to Google Drive:
   ```bash
   ping drive.google.com
   ```

### Issue 7: Job Queued Too Long / No GPU Available

**Solution:** Try alternative partitions or check availability:
```bash
sinfo  # Check available nodes
srun --partition=interactive-gpu --gres=gpu:1 --time=1:00:00 --pty bash
```

---

## Compiler Requirements

| Component | Requirement | Tested Version |
|---|---|---|
| NVIDIA CUDA Compiler | >= 11.0 | 12.2.2 |
| GCC Compiler | >= 5.x recommended | default (with module) |
| C++ Standard | C++11 or later | C++11 |
| Compiler Flags | `-O3 -std=c++11 -arch=sm_70` | Required for Tesla V100 |

---

## Project Files

| File | Purpose |
|------|---------|
| `main.cu` | Main program and evaluation framework |
| `matrix.cu` / `matrix.h` | Matrix I/O and utilities (CSR format) |
| `kernelCPU.cu` | CPU baseline SpTRSV solver |
| `kernel0.cu` - `kernel3.cu` | GPU kernel implementations |
| `common.h` | Shared data structures |
| `timer.h` | Performance timing utilities |
| `Makefile` | Build configuration |
| `README.md` | This file |
| `MILESTONE_2.md` | Detailed Milestone 2 results and analysis |
| `data/` | Sparse matrix datasets (CSR format) |

---

## Notes

- All times reported are in **milliseconds**
- Results use kernel 0 by default (specified with `-0` flag)
- Data files are in CSR (Compressed Sparse Row) sparse matrix format
- GPU speedup varies significantly based on problem size and matrix structure
- Floating-point precision differences between GPU and CPU are normal and expected

---

**Last Updated:** April 14, 2026  
**Author:** CMPS 324 Project  
**Repository:** https://github.com/lauremhsn/SpTRSV.git  
**Milestone 2 Status:** ✅ Complete - GPU compilation working, benchmarks collected
