// Please make sure the following five std headers are included before hnswlib.h
#include <chrono>      // REPLACED Timer.hpp WITH THIS STANDARD HEADER
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <iomanip> // For std::setprecision

#include "hnswlib/hnswlib.h"
#include "hnsw_global.h" // Assumes your read_config and hnsw_params are here

#include <Exception.h>   // Assumes this is part of your utility library
#include <StringUtils.hpp> // Assumes this is part of your utility library

using namespace StringUtils;
using namespace std;
using namespace hnswlib;


// Re-using the load_data function for query vectors (.fvecs format)
void load_data(const char *filename, std::vector<float> &data, unsigned &num,
               unsigned &dim) {
  std::ifstream in(filename, std::ios::binary);
  NPP_ENFORCE(in.is_open());

  unsigned r_dim;
  in.read((char *)&r_dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  
  if (dim == 0) dim = r_dim;
  if (num == 0) num = (unsigned)(fsize / (dim + 1) / 4);

  NPP_ENFORCE(r_dim == dim);
  data.resize((size_t)num * (size_t)dim);

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char *)(&data[i * dim]), dim * 4);
  }
  in.close();
}

// New function to write query results to an .ivecs file
void write_results_ivecs(const char* filename, const vector<vector<int>>& results, unsigned num_queries, unsigned k) {
    cout << "Writing " << num_queries << " query results to " << filename << "..." << endl;
    std::ofstream out(filename, std::ios::binary);
    NPP_ENFORCE(out.is_open());

    for (size_t i = 0; i < num_queries; ++i) {
        // Write the dimensionality of the result vector (which is k)
        out.write(reinterpret_cast<const char*>(&k), sizeof(unsigned));
        // Write the actual k nearest neighbor labels
        out.write(reinterpret_cast<const char*>(results[i].data()), k * sizeof(int));
    }
    out.close();
    cout << "Results successfully written." << endl;
}


void querying(
    const char* i_filename,   // index filename
    const char* q_filename,   // query dataset filename
    const char* res_filename, // result output filename
    unsigned num_queries,     // # of query points
    unsigned dim,             // dimension
    unsigned K,               // # of neighbors to find
    unsigned efSearch,        // parameter efSearch
    float p_value             // parameter p for lp metric
) {
    // 1. Load query data
    std::vector<float> queries;
    cout << "Loading query data from " << q_filename << "..." << endl;
    load_data(q_filename, queries, num_queries, dim);

    // 2. Create the correct space based on p_value
    hnswlib::SpaceInterface<float>* space = nullptr;
    if (p_value == 2.0f) {
        space = new hnswlib::L2Space(dim);
    } else if (p_value == 1.0f) {
        space = new hnswlib::L1Space(dim);
    } else if (p_value > 0.0f) {
        space = new hnswlib::LpSpace(dim, p_value);
    } else {
        std::cerr << "Error: Unknown metric parameter '" << p_value << "'." << std::endl;
        return;
    }

// 3. Load the HNSW index
    cout << "Loading index from " << i_filename << "..." << endl;
    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(space, i_filename, false);
    cout << "Index loaded. Max elements: " << hnsw->max_elements_ << endl;

    // 4. Set query-time parameter
    hnsw->setEf(efSearch);
    
    // 5. Perform queries sequentially (single-threaded)
    cout << "Performing " << num_queries << " queries with K=" << K << " and efSearch=" << efSearch << "..." << endl;
    vector<vector<int>> all_results(num_queries);
    
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_queries; ++i) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result_queue = hnsw->searchKnn(
            (void*)(&queries[i * dim]), K
        );
        
        all_results[i].resize(K);
        for (int j = 0; j < K; ++j) {
            all_results[i][K - 1 - j] = static_cast<int>(result_queue.top().second);
            result_queue.pop();
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    double total_time_ms = elapsed_ms.count();

    // 6. Write results to the specified .ivecs file
    write_results_ivecs(res_filename, all_results, num_queries, K);

    // 7. Report performance to console
    double qps = num_queries / (total_time_ms / 1000.0);
    cout << "\n----- Query Performance -----" << endl;
    cout << "Total time: " << total_time_ms << " ms" << endl;
    cout << "Queries Per Second (QPS): " << qps << endl;
    cout << "---------------------------" << endl;

    // --- BEGIN: NEW LOGGING FUNCTIONALITY ---
    // Open result.txt in append mode
    std::ofstream result_log("/media/gtnetuser/T7/Huayi/lp_graph/Experiments/result.txt", std::ios_base::app);
    if (result_log.is_open()) {
        // Write the query filename on the first line
        result_log << q_filename << std::endl;
        // Write p-value and total query time on the second line
        result_log << std::fixed << std::setprecision(4) << p_value << "," << total_time_ms << std::endl;
        // Add a blank line for better separation between entries
        result_log << std::endl; 
        result_log.close(); // Ensure data is flushed to the file
    } else {
        std::cerr << "Warning: Could not open result.txt for logging." << std::endl;
    }
    // --- END: NEW LOGGING FUNCTIONALITY ---

    delete hnsw;
    delete space;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: ./hnsw_query [config file]";
    return EXIT_FAILURE;
  }
  
  hnsw_params conf = read_config(argv[1]);

  char i_filename[200] = "";
  char q_filename[200] = "";
  char res_filename[200] = "";

  strcpy(i_filename, conf.i_filename.c_str());
  strcpy(q_filename, conf.q_filename.c_str());
  strcpy(res_filename, conf.res_filename.c_str());
  
  unsigned nQueries = conf.nq;
  unsigned pointsDimension = conf.d;
  unsigned K = conf.K;
  unsigned efSearch = conf.efS;
  float p_value = conf.p;

  querying(i_filename, q_filename, res_filename, nQueries, pointsDimension, K, efSearch, p_value);
  
  return EXIT_SUCCESS;
}