#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

#ifndef VERIFY_PREFETCH_DISTANCE
#define VERIFY_PREFETCH_DISTANCE 1
#endif

#ifndef VERIFY_DISABLE_PREFETCH
#define VERIFY_DISABLE_PREFETCH 0
#endif

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "Error: " << msg << std::endl;
    std::exit(1);
}

void load_fvecs(const char* filename, std::vector<float>& data, uint32_t& num, uint32_t& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        die(std::string("Could not open file: ") + filename);
    }

    uint32_t first_dim = 0;
    if (!in.read(reinterpret_cast<char*>(&first_dim), sizeof(first_dim))) {
        die(std::string("Failed to read first dimension from: ") + filename);
    }
    if (first_dim == 0) {
        die(std::string("Invalid zero dimension in: ") + filename);
    }

    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    const size_t record_size = sizeof(uint32_t) + static_cast<size_t>(first_dim) * sizeof(float);
    if (record_size == 0 || file_size % record_size != 0) {
        die(std::string("Corrupted .fvecs file (record-size mismatch): ") + filename);
    }

    const uint32_t file_num = static_cast<uint32_t>(file_size / record_size);
    if (dim == 0) {
        dim = first_dim;
    }
    if (dim != first_dim) {
        die(std::string("Dimension mismatch in .fvecs: ") + filename);
    }

    if (num == 0) {
        num = file_num;
    } else if (num != file_num) {
        die(std::string("Vector count mismatch in .fvecs: ") + filename);
    }

    data.resize(static_cast<size_t>(num) * static_cast<size_t>(dim));
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        uint32_t current_dim = 0;
        if (!in.read(reinterpret_cast<char*>(&current_dim), sizeof(current_dim))) {
            die(std::string("Failed to read vector dimension at index ") + std::to_string(i) + " from: " + filename);
        }
        if (current_dim != dim) {
            die(std::string("Irregular .fvecs dimension at index ") + std::to_string(i) + " in: " + filename);
        }
        if (!in.read(reinterpret_cast<char*>(data.data() + i * dim), static_cast<std::streamsize>(dim * sizeof(float)))) {
            die(std::string("Failed to read vector payload at index ") + std::to_string(i) + " from: " + filename);
        }
    }
}

void load_ivecs(const char* filename, std::vector<std::vector<int32_t>>& data, uint32_t& num, uint32_t& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        die(std::string("Could not open file: ") + filename);
    }

    uint32_t first_dim = 0;
    if (!in.read(reinterpret_cast<char*>(&first_dim), sizeof(first_dim))) {
        die(std::string("Failed to read first dimension from: ") + filename);
    }
    if (first_dim == 0) {
        die(std::string("Invalid zero dimension in: ") + filename);
    }

    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    const size_t record_size = sizeof(uint32_t) + static_cast<size_t>(first_dim) * sizeof(int32_t);
    if (record_size == 0 || file_size % record_size != 0) {
        die(std::string("Corrupted .ivecs file (record-size mismatch): ") + filename);
    }

    const uint32_t file_num = static_cast<uint32_t>(file_size / record_size);
    if (dim == 0) {
        dim = first_dim;
    }
    if (dim != first_dim) {
        die(std::string("Dimension mismatch in .ivecs: ") + filename);
    }

    if (num == 0) {
        num = file_num;
    } else if (num != file_num) {
        die(std::string("Vector count mismatch in .ivecs: ") + filename);
    }

    data.resize(num, std::vector<int32_t>(dim));
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        uint32_t current_dim = 0;
        if (!in.read(reinterpret_cast<char*>(&current_dim), sizeof(current_dim))) {
            die(std::string("Failed to read ivecs dimension at index ") + std::to_string(i) + " from: " + filename);
        }
        if (current_dim != dim) {
            die(std::string("Irregular .ivecs dimension at index ") + std::to_string(i) + " in: " + filename);
        }
        if (!in.read(reinterpret_cast<char*>(data[i].data()), static_cast<std::streamsize>(dim * sizeof(int32_t)))) {
            die(std::string("Failed to read ivecs payload at index ") + std::to_string(i) + " from: " + filename);
        }
    }
}

void write_ivecs(const char* filename, const std::vector<std::vector<int32_t>>& data, uint32_t dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        die(std::string("Could not open output file for writing: ") + filename);
    }

    const uint32_t row_dim = dim;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != row_dim) {
            die("Output row size mismatch while writing ivecs at row " + std::to_string(i));
        }
        out.write(reinterpret_cast<const char*>(&row_dim), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(data[i].data()),
                  static_cast<std::streamsize>(row_dim * sizeof(int32_t)));
        if (!out.good()) {
            die("Failed while writing ivecs output at row " + std::to_string(i));
        }
    }
}

bool valid_id(int32_t id, uint32_t n_data) {
    return id >= 0 && static_cast<uint32_t>(id) < n_data;
}

constexpr size_t kPrefetchDistance =
    (VERIFY_PREFETCH_DISTANCE > 0) ? static_cast<size_t>(VERIFY_PREFETCH_DISTANCE) : 0;

inline void prefetch_read(const float* ptr) {
#if VERIFY_DISABLE_PREFETCH
    (void)ptr;
#else
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);
#else
    (void)ptr;
#endif
#endif
}

inline void prefetch_candidate_vector(const std::vector<int32_t>& candidates, size_t lookahead_index,
                                      uint32_t n_data, uint32_t d_data, const std::vector<float>& data_vecs,
                                      uint32_t query_index) {
#if VERIFY_DISABLE_PREFETCH
    (void)candidates;
    (void)lookahead_index;
    (void)n_data;
    (void)d_data;
    (void)data_vecs;
    (void)query_index;
#else
    if (lookahead_index >= candidates.size()) {
        return;
    }
    const int32_t lookahead_id = candidates[lookahead_index];
    if (!valid_id(lookahead_id, n_data)) {
        die("Candidate ID out of range during prefetch at query " + std::to_string(query_index) +
            ", position " + std::to_string(lookahead_index) + ", id=" + std::to_string(lookahead_id));
    }
    const float* lookahead_vec = data_vecs.data() + static_cast<size_t>(lookahead_id) * d_data;
    prefetch_read(lookahead_vec);
#endif
}

struct Candidate {
    int32_t id;
    float dist;
    bool operator<(const Candidate& other) const { return dist < other.dist; }
};

size_t intersection_size_sorted(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    size_t i = 0;
    size_t j = 0;
    size_t overlap = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            ++i;
        } else if (a[i] > b[j]) {
            ++j;
        } else {
            ++overlap;
            ++i;
            ++j;
        }
    }
    return overlap;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 9) {
        std::cout << "Usage: ./candidate_verify [data_file] [query_file] [candidate_file] [p_value] [K] [batch_size] [tau] [output_file]" << std::endl;
        std::cout << "Example: ./candidate_verify sift_base.fvecs sift_query.fvecs candidates.ivecs 1.5 50 100 0.92 result.ivecs" << std::endl;
        return 1;
    }

    const char* data_file = argv[1];
    const char* query_file = argv[2];
    const char* cand_file = argv[3];
    const char* output_file = argv[8];

    float p_value = 0.0f;
    uint32_t K = 0;
    uint32_t batch_size = 0;
    float tau = 0.0f;
    try {
        p_value = std::stof(argv[4]);
        K = static_cast<uint32_t>(std::stoul(argv[5]));
        batch_size = static_cast<uint32_t>(std::stoul(argv[6]));
        tau = std::stof(argv[7]);
    } catch (const std::exception& e) {
        die(std::string("Failed to parse arguments: ") + e.what());
    }

    if (p_value <= 0.0f) {
        die("p_value must be > 0.");
    }
    if (K == 0) {
        die("K must be > 0.");
    }
    if (batch_size == 0) {
        die("batch_size must be > 0.");
    }
    if (tau < 0.0f || tau > 1.0f) {
        die("tau must be in [0, 1].");
    }

    std::cout << "--- Loading Data ---" << std::endl;
    uint32_t n_data = 0;
    uint32_t d_data = 0;
    std::vector<float> data_vecs;
    load_fvecs(data_file, data_vecs, n_data, d_data);
    std::cout << "Loaded Base Data: " << n_data << " vectors, dim=" << d_data << std::endl;

    uint32_t n_query = 0;
    uint32_t d_query = 0;
    std::vector<float> query_vecs;
    load_fvecs(query_file, query_vecs, n_query, d_query);
    std::cout << "Loaded Queries: " << n_query << " vectors, dim=" << d_query << std::endl;

    uint32_t n_cand = 0;
    uint32_t d_cand = 0;
    std::vector<std::vector<int32_t>> candidates;
    load_ivecs(cand_file, candidates, n_cand, d_cand);
    std::cout << "Loaded Candidates: " << n_cand << " sets, size=" << d_cand << std::endl;

    if (d_data != d_query) {
        die("Dimension mismatch between base vectors and queries.");
    }
    if (n_query == 0) {
        die("Query file is empty.");
    }
    if (n_cand != n_query) {
        die("Number of candidate sets must match number of queries.");
    }
    if (d_cand < K) {
        die("Candidate list length t is smaller than K. Stage-1 must output t >= K.");
    }
    if (d_cand == K) {
        std::cerr << "Warning: Candidate list length equals K. Verification will not refine beyond initialization." << std::endl;
    }

    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    if (std::abs(p_value - 2.0f) < 1e-6f) {
        space = std::make_unique<hnswlib::L2Space>(d_data);
    } else if (std::abs(p_value - 1.0f) < 1e-6f) {
        space = std::make_unique<hnswlib::L1Space>(d_data);
    } else {
        space = std::make_unique<hnswlib::LpSpace>(d_data, p_value);
    }

    const hnswlib::DISTFUNC<float> dist_func = space->get_dist_func();
    void* dist_func_param = space->get_dist_func_param();

    std::cout << "\n--- Starting Verification (p=" << p_value << ", K=" << K
              << ", batch=" << batch_size << ", tau=" << tau << ") ---" << std::endl;

    double total_time_ms = 0.0;
    double total_dists_computed = 0.0;
    std::vector<std::vector<int32_t>> final_results(n_query, std::vector<int32_t>(K));

    for (uint32_t qi = 0; qi < n_query; ++qi) {
        const float* q_vec = query_vecs.data() + static_cast<size_t>(qi) * d_query;
        const std::vector<int32_t>& curr_cands = candidates[qi];

        if (curr_cands.size() < K) {
            die("Encountered per-query candidate list with size < K at query index " + std::to_string(qi));
        }

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<Candidate> top_k;
        top_k.reserve(static_cast<size_t>(K) + batch_size);

        for (uint32_t j = 0; j < K; ++j) {
            if (kPrefetchDistance != 0) {
                const size_t lookahead = static_cast<size_t>(j) + kPrefetchDistance;
                if (lookahead < K) {
                    prefetch_candidate_vector(curr_cands, lookahead, n_data, d_data, data_vecs, qi);
                }
            }
            const int32_t id = curr_cands[j];
            if (!valid_id(id, n_data)) {
                die("Candidate ID out of range at query " + std::to_string(qi) +
                    ", position " + std::to_string(j) + ", id=" + std::to_string(id));
            }
            const float* d_vec = data_vecs.data() + static_cast<size_t>(id) * d_data;
            top_k.push_back({id, dist_func(q_vec, d_vec, dist_func_param)});
        }
        std::sort(top_k.begin(), top_k.end());
        total_dists_computed += static_cast<double>(K);

        std::vector<int32_t> old_top_k_ids;
        std::vector<int32_t> new_top_k_ids;
        old_top_k_ids.reserve(K);
        new_top_k_ids.reserve(K);

        size_t idx = K;
        while (idx < curr_cands.size()) {
            const size_t current_batch = std::min(static_cast<size_t>(batch_size), curr_cands.size() - idx);
            if (current_batch == 0) {
                break;
            }

            old_top_k_ids.clear();
            for (const Candidate& c : top_k) {
                old_top_k_ids.push_back(c.id);
            }
            std::sort(old_top_k_ids.begin(), old_top_k_ids.end());

            for (size_t j = 0; j < current_batch; ++j) {
                if (kPrefetchDistance != 0) {
                    const size_t lookahead = idx + j + kPrefetchDistance;
                    if (lookahead < curr_cands.size()) {
                        prefetch_candidate_vector(curr_cands, lookahead, n_data, d_data, data_vecs, qi);
                    }
                }
                const int32_t id = curr_cands[idx + j];
                if (!valid_id(id, n_data)) {
                    die("Candidate ID out of range at query " + std::to_string(qi) +
                        ", position " + std::to_string(idx + j) + ", id=" + std::to_string(id));
                }
                const float* d_vec = data_vecs.data() + static_cast<size_t>(id) * d_data;
                top_k.push_back({id, dist_func(q_vec, d_vec, dist_func_param)});
            }
            total_dists_computed += static_cast<double>(current_batch);

            std::sort(top_k.begin(), top_k.end());
            if (top_k.size() > K) {
                top_k.resize(K);
            }

            new_top_k_ids.clear();
            for (const Candidate& c : top_k) {
                new_top_k_ids.push_back(c.id);
            }
            std::sort(new_top_k_ids.begin(), new_top_k_ids.end());

            const size_t overlap = intersection_size_sorted(old_top_k_ids, new_top_k_ids);
            const float ratio = static_cast<float>(overlap) / static_cast<float>(K);
            idx += current_batch;
            if (ratio >= tau) {
                break;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time_ms += elapsed.count();

        for (uint32_t j = 0; j < K; ++j) {
            final_results[qi][j] = top_k[j].id;
        }
    }

    std::cout << "\n----- Verification Results -----" << std::endl;
    std::cout << "Total Verification Time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Avg Time per Query: " << (total_time_ms / static_cast<double>(n_query)) << " ms" << std::endl;
    std::cout << "Avg Dists Computed: " << (total_dists_computed / static_cast<double>(n_query)) << std::endl;
    std::cout << "--------------------------------" << std::endl;

    std::ofstream log("verification_log.txt", std::ios_base::app);
    if (log.is_open()) {
        log << std::fixed << std::setprecision(4) << p_value << "," << total_time_ms << std::endl;
    }

    write_ivecs(output_file, final_results, K);

    return 0;
}
