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

---

## Prerequisites

### Local Machine (Ran on Mac)
- Git
- SSH access to HPC cluster (AUB VPN required. SEE: https://servicedesk.aub.edu.lb/TDClient/1398/Portal/Requests/ServiceDet?ID=29740)
- Google Drive access for data download

### HPC Cluster Requirements
- CUDA toolkit (tested with 12.4.0)
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
│   ├── kernelCPU0.cu
│   ├── kernel0.cu
│   ├── kernel1.cu
│   ├── kernel2.cu
│   ├── kernel3.cu
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

# Download rajat18.txt (16 MB)
curl -L 'https://drive.google.com/uc?export=download&id=1nn6LGyE7IJjPR84D-tXx7lG1f-JaFJUe' -o data/rajat18.txt

# Download parabolic_fem.txt (69 MB)
curl -L 'https://drive.google.com/uc?export=download&id=1yoSNheQxMyzXIXwOyhUplBP7FZaty6o5' -o data/parabolic_fem.txt

# Download tmt_sym.txt (93 MB)
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
srun --partition=gpu --gres=gpu:1 --time=0:30:00 --pty bash
```

This allocates a GPU node (typically takes a few seconds). Your prompt will change to show the compute node name (e.g., `[rmd35@onode26 ~]$`).

**Note:** Use `sinfo` to check for availability.

### 2. Load CUDA Module

```bash
module load cuda/12.4.0
```

Verify the CUDA installation:

```bash
nvidia-smi
```

### 3. Compile the Code

The Makefile requires C++11 support. Compile with the appropriate flags:

```bash
cd ~/SpTRSV/sptrsv
make NVCC_FLAGS="-O3 -std=c++11"
```

If compilation is successful, you should see `sptrsv` executable created.

**Note about kernels:** The project includes:
- `kernelCPU0.cu` - CPU baseline implementation
- `kernel0.cu`, `kernel1.cu`, `kernel2.cu`, `kernel3.cu` - GPU kernel variants

The Makefile compiles all kernels by default. The `-d` flag at runtime selects which dataset to use, not which kernel.

---

## Running Evaluations

### Prerequisites Before Running

Ensure you're on a GPU node with:
- CUDA loaded: `module load cuda/12.4.0`
- Data files present: `ls data/` shows all three files
- Executable compiled: `ls sptrsv` exists

### Run the Evaluation

Execute the evaluation on all three datasets:

```bash
./sptrsv -d s   # Small dataset (rajat18)
./sptrsv -d m   # Medium dataset (parabolic_fem)
./sptrsv -d l   # Large dataset (tmt_sym)
```

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

## Results

### Sample Evaluation Results (CPU Baseline)

Running on AMD EPYC 7551 with 2x Tesla V100 GPUs:

```
Small Dataset (rajat18):
   CPU time(128 cols): 247.541994 ms
   CPU time(256 cols): 513.773978 ms
   CPU time(512 cols): 1121.227980 ms

Medium Dataset (parabolic_fem):
   CPU time(128 cols): 2596.752882 ms
   CPU time(256 cols): 5236.655235 ms
   CPU time(512 cols): 10717.782974 ms

Large Dataset (tmt_sym):
   CPU time(128 cols): 1587.352991 ms
   CPU time(256 cols): 3311.953068 ms
   CPU time(512 cols): 8808.150291 ms
```

### Output Format

The evaluation runs sparse triangular solves with three different tile sizes:
- 128 columns per tile
- 256 columns per tile
- 512 columns per tile

Times are reported in milliseconds for each configuration.

---

## Troubleshooting

### Issue: `nvcc: command not found`

**Solution:** Load CUDA module on compute node (not head node):
```bash
srun --partition=gpu --gres=gpu:1 --pty bash
module load cuda/12.4.0
```

This means you are running on the head node. Head nodes typically don't have CUDA drivers installed.

### Issue: Compilation Error - C++11 Support Required

**Solution:** Ensure you're using the correct NVCC flags:
```bash
make NVCC_FLAGS="-O3 -std=c++11"
```

### Issue: `Error: could not open file data/rajat18.txt`

**Solution:** Verify data files exist:
```bash
ls -la data/
```

Download missing files using the curl commands from the [Data Files](#data-files) section.

### Issue: Google Drive Download Fails

**Troubleshooting:**
1. Verify files are shared with "Anyone with the link" access
2. Try alternative download (if internet stable):
   ```bash
   wget --no-check-certificate 'https://drive.google.com/uc?export=download&confirm=NO_ANTIVIRUS&id=FILE_ID' -O data/filename.txt
   ```
3. Check network connectivity to Google Drive:
   ```bash
   ping drive.google.com
   ```

### Issue: Job Queued Too Long / No GPU Available

**Solution:** Try alternative partitions:
```bash
sinfo  # Check available nodes
srun --partition=interactive-gpu --gres=gpu:1 --time=1:00:00 --pty bash
srun --partition=cudadev --gres=gpu:1 --time=1:00:00 --pty bash
```

---

## Compiler Requirements

- **NVIDIA CUDA Compiler (nvcc):** >= 11.0
- **C++ Standard:** C++11 or later (required by matrix.cu)
- **Compiler Flags:** `-O3 -std=c++11`

---

## Project Files

| File | Purpose |
|------|---------|
| `main.cu` | Main program and evaluation framework |
| `matrix.cu` / `matrix.h` | Matrix I/O and utilities |
| `kernelCPU0.cu` | CPU baseline SpTRSV solver |
| `kernel0.cu` - `kernel3.cu` | GPU kernel implementations |
| `common.h` | Shared data structures |
| `timer.h` | Performance timing utilities |
| `Makefile` | Build configuration |
| `data/` | Sparse matrix datasets |

---

## References

- CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/
- NVIDIA GPU Performance Tuning: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Sparse Matrix Formats: CSR (Compressed Sparse Row) format used in this project

---

## Notes

- All times reported are in **milliseconds**
- Code compiles to CPU-only baseline by default (GPU kernels need to be selected in main.cu)
- Data files are in CSR sparse matrix format
- Project was developed and tested on HPC clusters with NVIDIA GPUs

---

**Last Updated:** April 2, 2026
**Author:** CMPS 324 Project
**Repository:** https://github.com/lauremhsn/SpTRSV.git
