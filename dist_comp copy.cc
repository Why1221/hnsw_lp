#include <iostream>
#include <vector>
#include <chrono>
#include <random>    // For generating high-quality random data
#include <numeric>   // For std::accumulate
#include <iomanip>   // For std::fixed and std::setprecision

#include "hnswlib/hnswlib.h"


// Please make sure the following five std headers are included before hnswlib.h
#include <cmath>
#include <fstream>
#include <queue>


#include <Exception.h>
#include <StringUtils.hpp>
#include <Timer.hpp>

using namespace StringUtils;
using namespace std;
using namespace hnswlib;


void load_data(const char *filename, std::vector<std::vector<float>> &data, unsigned &num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    NPP_ENFORCE(in.is_open());

    // --- Read dimension from the first 4 bytes of the file ---
    in.read(reinterpret_cast<char*>(&dim), sizeof(unsigned));

    // --- Determine the number of vectors from the file size ---
    in.seekg(0, std::ios::end);
    long long fsize = in.tellg();
    // Each record is (4-byte dimension int + dim * 4-byte float)
    num = static_cast<unsigned>(fsize / (sizeof(unsigned) + dim * sizeof(float)));

    // --- Resize the 2D vector to the correct dimensions ---
    // This creates 'num' rows, each being a vector of 'dim' floats.
    data.resize(num, std::vector<float>(dim));

    // --- Read the data vector by vector ---
    in.seekg(0, std::ios::beg); // Go back to the beginning of the file

    for (size_t i = 0; i < num; i++) {
        // We need to read and discard the 4-byte dimension integer for each vector.
        unsigned record_dim;
        in.read(reinterpret_cast<char*>(&record_dim), sizeof(unsigned));
        NPP_ENFORCE(record_dim == dim); // Sanity check

        // Read the actual vector data directly into the inner vector's buffer.
        in.read(reinterpret_cast<char*>(data[i].data()), dim * sizeof(float));
    }

    in.close();
}


/**
 * @brief A helper function to generate a vector with high-quality random floats.
 * @param vec The vector to fill with random data.
 * @param rng The Mersenne Twister random number generator engine.
 * @param dist The uniform real distribution to use.
 */
void generate_random_vector(std::vector<float>& vec, std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    for (float& val : vec) {
        val = dist(rng);
    }
}

int main() {
    // --- Benchmark Parameters ---

    std::cout << "--- Distance Function Benchmark ---" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    unsigned num_invocations = 10000000; // Number of times to invoke the distance function
    // --- Data Preparation ---
    string train_filename = "/media/gtnetuser/T7/Huayi/lp_graph/Experiments/sift/sift_train.fvecs";
    string query_filename = "/media/gtnetuser/T7/Huayi/lp_graph/Experiments/sift/sift_test.fvecs";
    unsigned t_num = 2000000; // Number of vectors in the dataset
    unsigned q_num = 1000;  // Number of query vectors
    unsigned dim = 128; // Dimension of each vector
    float p_value = 0.5f; // 'p' value for Lp distance
    std::vector<std::vector<float>> train;
    std::vector<std::vector<float>> query;

    load_data(train_filename.c_str(), train, t_num, dim);
    load_data(query_filename.c_str(), query, q_num, dim);

    // This accumulator is used to store results, preventing the compiler
    // from optimizing away the distance calculations.
    double result_accumulator = 0.0;

    // --- 1. Benchmark L2 Distance (Native HNSWlib SIMD Optimized) ---
    {
        hnswlib::L2Space l2_space(dim);
        auto dist_func = l2_space.get_dist_func();
        void* params = l2_space.get_dist_func_param();

        auto start_time = std::chrono::high_resolution_clock::now();
        for (long long i = 0; i < num_invocations; ++i) {
            result_accumulator += dist_func(train[131].data(), query[14].data(), params);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "1. L2 Distance (SIMD Optimized):" << std::endl;
        std::cout << "   Total time: " << elapsed_ms.count() << " ms" << std::endl;
        std::cout << "Final accumulator value (to prevent optimization): " << result_accumulator << std::endl;
        std::cout << "   Time per call: " << elapsed_ms.count() * 1000 / num_invocations << " microseconds" << std::endl << std::endl;
    }

     result_accumulator = 0.0;
    // --- 2. Benchmark L1 Distance (Our Custom SIMD Optimized Version) ---
    {
        hnswlib::L1Space l1_space(dim);
        auto dist_func = l1_space.get_dist_func();
        void* params = l1_space.get_dist_func_param();


        auto start_time = std::chrono::high_resolution_clock::now();
        for (long long i = 0; i < num_invocations; ++i) {
            result_accumulator += dist_func(train[13114].data(), query[10].data(), params);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;

        std::cout << "2. L1 Distance (SIMD Optimized):" << std::endl;
        std::cout << "   Total time: " << elapsed_ms.count() << " ms" << std::endl;
        std::cout << "Final accumulator value (to prevent optimization): " << result_accumulator << std::endl;
        std::cout << "   Time per call: " << elapsed_ms.count() * 1000 / num_invocations << " microseconds" << std::endl << std::endl;
    }

    result_accumulator = 0.0;
    // --- 3. Benchmark Lp Distance (Our Custom Look-Up Table Optimized Version) ---
    {
        hnswlib::LpSpace lp_space(dim, p_value);
        auto dist_func = lp_space.get_dist_func();
        void* params = lp_space.get_dist_func_param();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        for (long long i = 0; i < num_invocations; ++i) {
            result_accumulator += dist_func(train[13114].data(), query[10].data(), params);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;

        std::cout << "3. Lp Distance (p=" << p_value << "):" << std::endl;
        std::cout << "   Total time: " << elapsed_ms.count() << " ms" << std::endl;
        std::cout << "Final accumulator value (to prevent optimization): " << result_accumulator << std::endl;
        std::cout << "   Time per call: " << elapsed_ms.count() * 1000 / num_invocations << " microseconds" << std::endl << std::endl;
    }
    
    // // This line can be uncommented for debugging to ensure the accumulator is used.
    // std::cout << "Final accumulator value (to prevent optimization): " << result_accumulator << std::endl;

    return 0;
}