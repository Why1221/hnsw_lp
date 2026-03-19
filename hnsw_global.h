#ifndef __HNSW_GLOBAL_H__
#define __HNSW_GLOBAL_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "hnswlib/json.hpp"
#include <fstream>
#include <algorithm>
#include <memory>
#include <random>
#include <cmath>
#include <limits>

using json = nlohmann::json;

struct hnsw_params
  {
    unsigned n;                    // Number of points in the dataset
    unsigned nq;                   // Number of query points
    unsigned d;                    // Dimension of the input points
    std::string ds_filename = "", i_filename = ""; // Dataset and index file paths
    std::string q_filename = "", res_filename = ""; // Query and result file paths
    float p;                  // Metric parameter p for Lp distance
    unsigned efS;                   // Size of the dynamic list for the nearest neighbors (used during the search). Higher values lead to better accuracy at the expense of slower search time.
    unsigned K;                    // Number of nearest neighbors to retrieve during querying.
    unsigned M;                    // Number of bi-directional links created for every new element during construction. Recommended range: 5-48. Higher values lead to better accuracy at the expense of increased index size and slower construction.
    unsigned efC;       // Size of the dynamic list for the nearest neighbors (used during the construction). Recommended range: 100-2000. Higher values lead to better accuracy at the expense of slower construction time.
  };

hnsw_params read_config(const char *filename)
  {
    std::ifstream json_f(filename);
    json config;
    json_f >> config;

    hnsw_params params;

    params.n = config.at("n");
    params.d = config.at("d");
    params.nq = config.at("nq");
    params.ds_filename = config.at("ds");
    params.i_filename = config.at("if");
    params.q_filename = config.at("qf");
    params.res_filename = config.at("rf");
    params.K = config.at("K");
    params.efS = config.at("efS");
    params.p = config.at("p");
    params.M = config.at("M");
    params.efC = config.at("Ef");

    return params;
  }

#endif // __HNSW_GLOBAL_H__