# U-HNSW: An Efficient Graph-based Solution to ANNS Under Universal Lp Metrics

This repository provides the C++ implementation of **U-HNSW**, the first graph-based indexing solution designed for Approximate Nearest Neighbor Search (ANNS) under universal $L_p$ metrics ($0 < p \le 2$).

## Prerequisites

* **Compiler**: GCC 7+ (must support C++17)
* **Libraries**: OpenMP
* **Hardware**: CPU supporting AVX2/AVX-512 instruction sets (required for SIMD optimizations)
* **Python**: Python 3.x (required for running experiment scripts)

## Compilation

The project uses a `Makefile` to build the core C++ executables. To compile all components, run:

```bash
make all
```
This will generate four executables in the root directory:
* `hnsw_index`: For building the graph index ($G_1$ or $G_2$).
* `hnsw_query`: For running query search on a selected graph index.
* `candidate_verify`: For Stage-2 candidate verification/refinement under target $L_p$.
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
* 1. Edit `scripts/run_index.py` to set dataset paths (`exp_path_org`) and index parameters (e.g., `M`, `efc`, `p`).
* 2. Run the script:
```bash
python scripts/run_index.py
```
This script will automatically generate the necessary JSON configuration files and invoke the `hnsw_index` executable to build and save the index to disk.

### 3. Query Processing
The query entrypoint is:

```bash
python scripts/run_query.py
```

`run_query.py` is parameterized by editing values at the top of the script (no CLI arguments). It uses JSON configuration files under `Experiments/{dataset}/config/` and supports two execution modes:

* **One-stage mode** (`p == 1.0` or `p == 2.0`):
  * Runs `hnsw_query` directly with final `K`.
  * Produces final output `.ivecs` file.

* **Two-stage mode** (`p != 1.0` and `p != 2.0`):
  * Stage 1 (Candidate Generation): run `hnsw_query` to generate `t` candidates.
  * Stage 2 (Candidate Verification): run `candidate_verify` to refine candidates to final top-`K`.

Parameter constraints enforced by the script:
* `K > 0`, `efs > 0`, `p > 0`.
* For two-stage mode: `t > 10` and `t > 4K`.

Stage-2 defaults:
* `batch_size = K`
* `tau = 0.92`

Output naming convention:
* Final output (always):  
  `{dataset}_{p}_{K}.ivecs`
* In two-stage mode, the Stage-1 candidate file is an intermediate artifact; the final output file is produced by `candidate_verify`.

`candidate_verify` interface note:
* `candidate_verify` requires an explicit output file argument:
```bash
./candidate_verify [data_file] [query_file] [candidate_file] [p_value] [K] [batch_size] [tau] [output_file]
```

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
