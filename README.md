# U-HNSW: An Efficient Graph-based Solution to ANNS Under Universal Lp Metrics

> **Research Prototype Note:** This repository contains the reference implementation of U-HNSW. The code is currently being refactored for better readability and ease of use. A fully polished version with comprehensive documentation will be finalized shortly.

This repository provides the C++ implementation of **U-HNSW**, the first graph-based indexing solution designed for Approximate Nearest Neighbor Search (ANNS) under universal $L_p$ metrics ($0 < p \le 2$).

## Prerequisites

* **Compiler**: GCC 7+ (must support C++17)
* **Libraries**: OpenMP
* **Hardware**: CPU supporting AVX2/AVX-512 instruction sets (required for SIMD optimizations)
* **Python**: Python 3.x (required for running experiment scripts)
* **Dependencies**: `numpy` (for Python scripts)

## Compilation

The project uses a `Makefile` to build the core C++ executables. To compile all components, run:

```bash
make all
```
This will generate three executables in the root directory:
* `hnsw_index`: For building the graph index ($G_1$ or $G_2$).
* `hnsw_query`: For running search queries (Candidate Generation).
* `dist_comp_time`: For benchmarking distance computation costs.

To clean up compiled files:
```bash
make clean
```

## Usage
### 1. Distance Computation Benchmark
To reproduce the analysis of computational costs for different $L_p$ metrics (as discussed in the "Background" section of the paper), run the benchmark tool:
```bash
./dist_comp_time
```

This program measures and compares the execution time of $L_1$, $L_2$, and general $L_p$ distance computations using the implemented SIMD optimizations.

### 2. Index Construction
We provide Python wrapper scripts in the `scripts/` directory to handle configuration generation and index construction.

To build the base HNSW indices ($G_1$ optimized for $L_1$, or $G_2$ optimized for $L_2$):
* 1.Edit `scripts/run_index.py` to set your specific dataset paths (`exp_path_org`) and index parameters (e.g., `M`, `efc`, `p`).
* 2.Run the script:
```bash
python scripts/run_index.py
```
This script will automatically generate the necessary JSON configuration files and invoke the `hnsw_index` executable to build and save the index to disk.

### 3. Query Processing
To perform the nearest neighbor search (currently covering the Candidate Generation phase):
* 1.Edit `scripts/run_query_k.py` to set your dataset paths, the query metric $p$, and search parameters (e.g., $K$, $efS$).

* 2. Run the script:
```bash
python scripts/run_query_k.py
```
This script executes the query on the constructed index and outputs the candidate results to disk (in `.ivecs` format).

## Datasets

Some of the datasets used in the paper can be found here:
[Test Datasets in Zenodo](https://zenodo.org/records/18626438?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjBiMjhmYTU0LWI2MzktNDJiOC04OThiLWIzNmNjN2ExYzcxYyIsImRhdGEiOnt9LCJyYW5kb20iOiI2ZmQxMmU4ZDYyMTBlNGI3OTM4NTM4NWU5OGVjMTIzOCJ9.4fEdjRrNccXuXL-tyB7kTnyRxDqr9TDsyhO-WouD4MMt8Oc2nhQrnDnk6FuQ15yGJBQgTi7pH3N5SQ6KCSZqzQ)

## References

* HNSW paper:

```Latex
@article{malkov2018efficient,
  title={Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs},
  author={Malkov, Yu A and Yashunin, Dmitry A},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={42},
  number={4},
  pages={824--836},
  year={2018},
  publisher={IEEE}
}
```
* NMSLIB
[NMSLIB Github](https://github.com/nmslib/nmslib)