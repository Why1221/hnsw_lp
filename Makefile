# Makefile for HNSWlib-based projects
#
# Usage:
#   make                - Compiles all applications (build_app, query_app, benchmark_distances)
#   make build_app      - Compiles only the index building application
#   make query_app      - Compiles only the index querying application
#   make benchmark      - Compiles only the distance benchmark application
#   make clean          - Removes all compiled executables

# --- Compiler and Flags ---

# Compiler to use
CXX = g++

# C++ standard and optimization flags.
# -std=c++17: Use the C++17 standard
# -O3:        Enable the highest level of optimization
# -pthread:   Enable thread support
# -mavx -mavx2: Enable AVX/AVX2 instruction sets for SIMD
CXXFLAGS = -std=c++17 -O3 -pthread -mavx -mavx2 -msse4.2 -mavx512f -I Common -march=native -fpic

# Linker flags.
# -fopenmp: Link with the OpenMP library for parallel processing
LDFLAGS = -fopenmp

# --- Source Files ---

# Common HNSWlib source file required by all applications
HNSWLIB_SRC = hnswlib/hnswlib.h

# Source files for each application
BUILD_SRC = hnsw_index.cc
QUERY_SRC = hnsw_query.cc
BENCHMARK_SRC = dist_comp.cc

# --- Executable Names ---

BUILD_APP = hnsw_index
QUERY_APP = hnsw_query
BENCHMARK_APP = dist_comp_time

# --- Targets ---

# The default target, executed when you just run `make`.
# It depends on all application targets.
all: $(BUILD_APP) $(QUERY_APP) $(BENCHMARK_APP)

# Rule to build the index building application.
# It depends on its source file and the HNSWlib source file.
$(BUILD_APP): $(BUILD_SRC) $(HNSWLIB_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "--- Built '$@' successfully. ---"

# Rule to build the index querying application.
$(QUERY_APP): $(QUERY_SRC) $(HNSWLIB_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ 
	@echo "--- Built '$@' successfully. ---"
	
# Rule to build the distance benchmark application.
# Renaming the target to 'benchmark' for easier typing.
$(BENCHMARK_APP): $(BENCHMARK_SRC) $(HNSWLIB_SRC)
	$(CXX) $(CXXFLAGS) $(BENCHMARK_SRC) $(HNSWLIB_SRC) -o $(BENCHMARK_APP) $(LDFLAGS)
	@echo "--- Built '$(BENCHMARK_APP)' successfully. Use 'make benchmark' to build. ---"


# The "clean" target to remove compiled files.
clean:
	@echo "Cleaning up compiled files..."
	rm -f $(BUILD_APP) $(QUERY_APP) $(BENCHMARK_APP)

# Declare targets that are not actual files as "PHONY".
# This prevents `make` from getting confused if a file named "all" or "clean" exists.
.PHONY: all clean benchmark