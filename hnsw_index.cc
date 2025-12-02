// Please make sure the following five std headers are included before hnswlib.h
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <omp.h> 

#include "hnswlib/hnswlib.h"
#include "hnsw_global.h"

#include <Exception.h>
#include <StringUtils.hpp>
#include <Timer.hpp>

using namespace StringUtils;
using namespace std;
using namespace hnswlib;


void load_data(const char *filename, std::vector<float> &data, unsigned &num,
               unsigned &dim) { // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  // if (!in.is_open()) {
  //   std::cout << "open file error" << std::endl;
  //   exit(-1);
  // }
  NPP_ENFORCE(in.is_open());

  unsigned r_dim, r_n;
  in.read((char *)&r_dim, 4);

  NPP_ENFORCE(r_dim == dim);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  r_n = (unsigned)(fsize / (dim + 1) / 4);

  NPP_ENFORCE(r_n == num);
  data.resize((size_t)num * (size_t)dim);

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char *)(&data[i * dim]), dim * 4);
  }
  in.close();
}



void indexing(const char *ds_filename, // database filename
              const char *i_filename,  // index filename (output)
              unsigned num,            // # of points
              unsigned dim,            // dimension
              unsigned M,              // parameter M
              unsigned efConstruction, // parameter efConstruction
              float p_value            // parameter p for lp metric
) {
  std::vector<float> train;
  load_data(ds_filename, train, num, dim);

  HierarchicalNSW<float> *appr_alg = nullptr;

  std::string perf_filename = join(
      {"hnsw-indexing", "n" + std::to_string(num), "d" + std::to_string(dim)},
      "-");
  perf_filename += ".txt";

  {
    std::string index_filename(i_filename);
    std::size_t found = index_filename.rfind("/");
    if (found != std::string::npos) {
      perf_filename = index_filename.substr(0, found) + "/" + perf_filename;
    }
  }

  hnswlib::SpaceInterface<float>* space = nullptr;
  if (p_value == 2.0f)
  {
    space = new hnswlib::L2Space(dim);
  } else if (p_value == 1.0f)
  {
    space = new hnswlib::L1Space(dim);
  } else if (p_value >0.1f && p_value < 2.0f)
  {
    space = new hnswlib::LpSpace(dim, p_value);
  } else {
        std::cerr << "Error: Unknown metric parameter  '" << p_value << "'." << std::endl;
        return;
    }


    hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(space, num, M, efConstruction);

    #pragma omp parallel for
  for (auto i = 0; i < num; ++i) {
    hnsw->addPoint((void *)(&train[i * dim]), (size_t)i);
  }

  hnsw->saveIndex(i_filename);

delete hnsw;
delete space;
}


int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: ./falconn_recall_tabel [config file]";
  }
  
  hnsw_params conf = read_config(argv[1]);

  char ds[200] = "";  // the file path of dataset
  char inf[200] = "";  // the folder path of index
  strcpy(ds, conf.ds_filename.c_str());
  strcpy(inf, conf.i_filename.c_str());
  unsigned nPoints = conf.n;
  unsigned pointsDimension = conf.d; // the dimensionality of points
  unsigned M = conf.M;
  unsigned efConstruction = conf.efC;
  float p_value = conf.p; // parameter p for lp metric


  indexing(ds, inf, nPoints, pointsDimension, M, efConstruction,p_value);
  return EXIT_SUCCESS;
}