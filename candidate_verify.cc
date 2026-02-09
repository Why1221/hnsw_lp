#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <unordered_set>

#include "hnswlib/hnswlib.h"

using namespace std;

// --- 1. Data Loading Utilities ---

// Load .fvecs (vectors of floats) - for Data and Queries
void load_fvecs(const char* filename, std::vector<float>& data, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }

    unsigned r_dim;
    in.read((char*)&r_dim, 4);
    in.seekg(0, std::ios::end);
    size_t fsize = (size_t)in.tellg();

    if (dim == 0) dim = r_dim;
    if (num == 0) num = (unsigned)(fsize / (dim + 1) / 4);

    if (r_dim != dim) {
        std::cerr << "Error: Dimension mismatch in " << filename << std::endl;
        exit(1);
    }

    data.resize((size_t)num * (size_t)dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(&data[i * dim]), dim * 4);
    }
    in.close();
}

// Load .ivecs (vectors of ints) - for Candidate Sets
void load_ivecs(const char* filename, std::vector<std::vector<int>>& data, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }

    // Read first dimension to verify
    unsigned r_dim;
    in.read((char*)&r_dim, 4);
    in.seekg(0, std::ios::end);
    size_t fsize = (size_t)in.tellg();
    
    num = (unsigned)(fsize / (r_dim * 4 + 4)); // Calculate number of vectors
    dim = r_dim;

    data.resize(num, std::vector<int>(dim));
    
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        unsigned d;
        in.read((char*)&d, 4); // Read dimension of this vector
        if (d != dim) {
            std::cerr << "Error: Irregular dimension in ivecs file at index " << i << std::endl;
            exit(1);
        }
        in.read((char*)data[i].data(), dim * 4);
    }
    in.close();
}

// --- 2. Helper Structs for Verification ---

struct Candidate {
    int id;
    float dist;
    
    // Sort ascending by distance
    bool operator<(const Candidate& other) const {
        return dist < other.dist;
    }
};

// --- 3. Core Verification Logic ---

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cout << "Usage: ./verify_candidates [data_file] [query_file] [candidate_file] [p_value] [K] [batch_size] [tau]" << std::endl;
        std::cout << "Example: ./verify_candidates sift_base.fvecs sift_query.fvecs candidates.ivecs 1.5 50 100 0.92" << std::endl;
        return 1;
    }

    const char* data_file = argv[1];
    const char* query_file = argv[2];
    const char* cand_file = argv[3];
    float p_value = std::stof(argv[4]);
    unsigned K = std::stoul(argv[5]);
    unsigned batch_size = std::stoul(argv[6]); // kappa
    float tau = std::stof(argv[7]);            // early termination threshold

    // --- Step A: Load All Data into Memory (Not timed) ---
    std::cout << "--- Loading Data ---" << std::endl;
    
    unsigned n_data, d_data = 0;
    std::vector<float> data_vecs;
    load_fvecs(data_file, data_vecs, n_data, d_data);
    std::cout << "Loaded Base Data: " << n_data << " vectors, dim=" << d_data << std::endl;

    unsigned n_query, d_query = 0;
    std::vector<float> query_vecs;
    load_fvecs(query_file, query_vecs, n_query, d_query);
    std::cout << "Loaded Queries: " << n_query << " vectors, dim=" << d_query << std::endl;

    unsigned n_cand, d_cand = 0;
    std::vector<std::vector<int>> candidates;
    load_ivecs(cand_file, candidates, n_cand, d_cand);
    std::cout << "Loaded Candidates: " << n_cand << " sets, size=" << d_cand << std::endl;

    if (d_data != d_query) {
        std::cerr << "Error: Dimension mismatch between data and query." << std::endl;
        return 1;
    }
    if (n_cand != n_query) {
        std::cerr << "Error: Number of candidate sets must match number of queries." << std::endl;
        return 1;
    }

    // --- Step B: Setup Distance Function (SIMD Optimized) ---
    hnswlib::SpaceInterface<float>* space = nullptr;
    if (std::abs(p_value - 2.0f) < 1e-6) space = new hnswlib::L2Space(d_data);
    else if (std::abs(p_value - 1.0f) < 1e-6) space = new hnswlib::L1Space(d_data);
    else space = new hnswlib::LpSpace(d_data, p_value);

    hnswlib::DISTFUNC<float> dist_func = space->get_dist_func();
    void* dist_func_param = space->get_dist_func_param();

    // --- Step C: Run Verification (TIMED) ---
    std::cout << "\n--- Starting Verification (p=" << p_value << ", K=" << K << ", batch=" << batch_size << ", tau=" << tau << ") ---" << std::endl;

    double total_time_ms = 0.0;
    double total_dists_computed = 0.0; // Statistical info

    for (unsigned i = 0; i < n_query; ++i) {
        const float* q_vec = &query_vecs[i * d_query];
        const std::vector<int>& curr_cands = candidates[i]; // The t candidates from Step 1
        
        // Safety check
        if (curr_cands.size() < K) {
            std::cerr << "Warning: Candidate set size (" << curr_cands.size() << ") smaller than K (" << K << ")." << std::endl;
            continue; 
        }

        // START TIMER (Per Query)
        auto start = std::chrono::high_resolution_clock::now();

        // 1. Initial R: First K candidates
        std::vector<Candidate> R;
        R.reserve(curr_cands.size()); // Reserve max possible to avoid realloc

        // Compute distances for initial K
        for(unsigned j=0; j<K; ++j) {
            int id = curr_cands[j];
            const float* d_vec = &data_vecs[id * d_data];
            float d = dist_func(q_vec, d_vec, dist_func_param);
            R.push_back({id, d});
        }
        std::sort(R.begin(), R.end()); // R is now sorted by Lp distance
        total_dists_computed += K;

        // 2. Iterative Verification
        unsigned idx = K;
        while(idx < curr_cands.size()) {
            // Determine batch size (handle end of list)
            unsigned current_batch = std::min(batch_size, (unsigned)curr_cands.size() - idx);
            if (current_batch == 0) break;

            // Process Batch
            std::vector<Candidate> batch_results;
            batch_results.reserve(current_batch);
            
            for(unsigned j=0; j<current_batch; ++j) {
                int id = curr_cands[idx + j];
                const float* d_vec = &data_vecs[id * d_data];
                float d = dist_func(q_vec, d_vec, dist_func_param);
                batch_results.push_back({id, d});
            }
            total_dists_computed += current_batch;

            // Merge R and batch_results -> R_new (Keep top K)
            // Strategy: Add batch to R, Sort, Resize to K
            R.insert(R.end(), batch_results.begin(), batch_results.end());
            std::sort(R.begin(), R.end()); 
            
            // Construct R_new (The new top K)
            // Actually, after sort, R[0...K-1] IS R_new.
            // We just need to check how many of these IDs were in the OLD R (before this batch).
            // But wait, the "Old R" is simply the top K from the previous iteration.
            
            // Wait, to calculate intersection efficiently:
            // Let R_old_ids be the set of IDs in R before adding the batch.
            // Since we sorted R in place, we need to be careful.
            
            // Optimized Intersection Check:
            // Since we only care about the Top-K.
            // We need to count: How many items in R[0...K] came from the Batch?
            // If very few came from the batch, it means R didn't change much.
            
            // Definition of stability: |R_new \cap R_old| / K >= tau
            // This is equivalent to: (K - |Items from Batch in Top K|) / K >= tau
            
            int items_from_batch_in_top_k = 0;
            // The batch items are those we just added. We can check IDs or just check logic.
            // To be strict:
            // We need to know which items in the new sorted R[0...K] were present in the previous R[0...K].
            // A simple way is to use a set or mark them.
            
            // Let's rely on IDs for correctness.
            // But doing a set intersection every time is slow.
            // Heuristic: If the worst item in Batch is better than best item in R, all replace.
            
            // Let's do the standard check as per algorithm description:
            // We need R_new (current top K) and R (previous top K).
            // We can reconstruct this check:
            // Intersection = K - (Number of elements in current top K that were NOT in previous top K)
            
            // Since we just merged Batch into R and resorted:
            // Any element in R[0...K] that came from 'batch_results' is "new".
            // Any element that came from the old 'R' is "old".
            
            // We can track this by checking if the ID exists in the batch set.
            // Optimization: put batch IDs in a small vector/set for checking.
            
            int overlap_count = 0;
            // To properly count overlap with PREVIOUS R (which had size K):
            // We need to know how many of the NEW Top K are from the OLD Top K.
            
            // Let's mark the candidates. 
            // Or simpler: Just keep a copy of old IDs? Copying K ints is cheap.
            std::vector<int> R_old_ids; 
            R_old_ids.reserve(K);
            // Before resizing/sorting, R (size K) was the old R.
            // Wait, I inserted batch into R and then sorted. 
            // So I need to save R's state before insert.
            
            // Refined Loop Logic:
            // 1. R contains current Top K.
            // 2. Save R's IDs to R_old_ids.
            // 3. Insert Batch. Sort. Resize to K.
            // 4. Count intersection(R, R_old_ids).
            
            // Since R is sorted by distance, not ID, we need standard intersection.
            
            // CORRECT IMPLEMENTATION:
            // 1. Snapshot Old R IDs.
            std::vector<int> old_top_k_ids;
            old_top_k_ids.reserve(K);
            for(int k=0; k<K; ++k) old_top_k_ids.push_back(R[k].id);
            std::sort(old_top_k_ids.begin(), old_top_k_ids.end()); // Sort by ID for std::intersection

            // 2. Sort current full R (Old + Batch) by distance
            // (Already did R.insert and sort logic above? No, let's fix flow)
            
            // FIX: The flow inside loop
            // R currently has size K (from init or previous loop)
            // Store Old R for check
            
            // Insert Batch
            // R is now size K + batch
            // Sort by distance
            // Keep top K
            if (R.size() > K) R.resize(K);

            // 3. Calculate Intersection
            std::vector<int> new_top_k_ids;
            new_top_k_ids.reserve(K);
            for(const auto& cand : R) new_top_k_ids.push_back(cand.id);
            std::sort(new_top_k_ids.begin(), new_top_k_ids.end()); // Sort by ID

            std::vector<int> intersection;
            std::set_intersection(old_top_k_ids.begin(), old_top_k_ids.end(),
                                  new_top_k_ids.begin(), new_top_k_ids.end(),
                                  std::back_inserter(intersection));
            
            float ratio = (float)intersection.size() / (float)K;
            
            if (ratio >= tau) {
                break; // Early termination
            }

            idx += current_batch;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time_ms += elapsed.count();
    }

    std::cout << "\n----- Verification Results -----" << endl;
    std::cout << "Total Verification Time: " << total_time_ms << " ms" << endl;
    std::cout << "Avg Time per Query: " << total_time_ms / n_query << " ms" << endl;
    std::cout << "Avg Dists Computed: " << total_dists_computed / n_query << endl;
    std::cout << "--------------------------------" << endl;

    // Optional: Write log for your paper aggregation
    std::ofstream log("verification_log.txt", std::ios_base::app);
    if(log.is_open()) {
        log << p_value << "," << total_time_ms << std::endl;
    }

    delete space;
    return 0;
}