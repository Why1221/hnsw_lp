#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include <iomanip>
#include <string>
#include <sstream>

// Include your custom LpSpace and the original L2Space
#include "hnswlib/hnswlib.h"

/**
 * @brief A generic function to benchmark any HNSWlib distance function using a pool of vectors.
 * @param name The name of the benchmark to print.
 * @param dist_func A pointer to the distance function to be tested.
 * @param params A void pointer to the function's parameters (usually the dimension).
 * @param data_pool A pool of pre-generated vectors to use for the benchmark.
 * @param num_invocations Number of times to call the function.
 */
void benchmark_distance(
    const std::string& name,
    hnswlib::DISTFUNC<float> dist_func,
    void* params,
    const std::vector<std::vector<float>>& data_pool,
    size_t num_invocations)
{
    double result_accumulator = 0.0;
    const size_t pool_size = data_pool.size();
    if (pool_size < 2) {
        std::cerr << "Error: Data pool must contain at least 2 vectors." << std::endl;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // The core loop now cycles through the vector pool
    for (size_t i = 0; i < num_invocations; ++i) {
        // Pick two different vectors from the pool for each iteration
        const auto& vec1 = data_pool[i % pool_size];
        const auto& vec2 = data_pool[(i + 1) % pool_size];
        result_accumulator += dist_func(vec1.data(), vec2.data(), params);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    
    std::cout << std::left << std::setw(35) << name
              << std::fixed << std::setprecision(4)
              << "Total: " << std::setw(12) << elapsed_ms.count() << "ms | "
              << "Per Call: " << (elapsed_ms.count() * 1000.0 / num_invocations) << " us" 
              << std::endl;
    
    // Using a volatile variable is a robust way to ensure the accumulator is used
    volatile double sink = result_accumulator;
}


int main() {
    // --- Benchmark Parameters ---
    const size_t num_invocations = 5000000; // Reduced for faster testing across many dims
    const size_t num_vectors_in_pool = 100; 

    // --- Lists to Iterate Over ---
    const std::vector<size_t> dims_to_test = {192,784,960,4096};
    const std::vector<float> p_values_to_test = {
        0.5f, 1.5f
    };

    std::cout << "--- Distance Function Benchmark ---" << std::endl;
    std::cout << "Invocations per test: " << num_invocations 
              << ", Vector Pool Size: " << num_vectors_in_pool << std::endl;

    // --- BEGIN: Outer loop for dimensions ---
    for (size_t current_dim : dims_to_test) {
        std::cout << "\n========================================================================" << std::endl;
        std::cout << "===== BENCHMARKING DIMENSION: " << current_dim << " =====" << std::endl;
        std::cout << "========================================================================" << std::endl;

        // --- Data Preparation for the current dimension ---
        std::cout << "Generating random data for dim=" << current_dim << "..." << std::endl;
        std::mt19937 rng(static_cast<unsigned int>(current_dim)); // Seed with dim for variety
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        std::vector<std::vector<float>> data_pool(num_vectors_in_pool, std::vector<float>(current_dim));
        for (size_t i = 0; i < num_vectors_in_pool; ++i) {
            for (float& val : data_pool[i]) {
                val = dist(rng);
            }
        }
        std::cout << "Data generation complete.\n" << std::endl;
        
        // --- Baselines ---
        std::cout << "--- Baselines for dim=" << current_dim << " ---" << std::endl;
        {
            hnswlib::L2Space native_l2_space(current_dim);
            benchmark_distance(
                "HNSWlib Native L2Sqr",
                native_l2_space.get_dist_func(),
                native_l2_space.get_dist_func_param(),
                data_pool, num_invocations
            );
        }
        {
            hnswlib::L1Space custom_l1_space(current_dim);
            benchmark_distance(
                "Custom L1",
                custom_l1_space.get_dist_func(),
                custom_l1_space.get_dist_func_param(),
                data_pool, num_invocations
            );
        }
        std::cout << "------------------------------------------------------------------------" << std::endl;
        
        // --- Custom LpSpace Benchmarks ---
        std::cout << "--- Custom LpSpace Benchmarks for dim=" << current_dim << " ---" << std::endl;
        for (float p_val : p_values_to_test) {
            std::stringstream ss;
            ss << "Custom LpSpace (p=" << std::fixed << std::setprecision(2) << p_val << ")";
            std::string benchmark_name = ss.str();
            
            try {
                hnswlib::LpSpace lp_space(current_dim, p_val);
                benchmark_distance(
                    benchmark_name,
                    lp_space.get_dist_func(),
                    lp_space.get_dist_func_param(),
                    data_pool, num_invocations
                );
            } catch (const std::exception& e) {
                std::cerr << "Error benchmarking p=" << p_val << ": " << e.what() << std::endl;
            }
        }
    } // --- END: Outer loop for dimensions ---
    std::cout << "\nBenchmark finished." << std::endl;

    return 0;
}