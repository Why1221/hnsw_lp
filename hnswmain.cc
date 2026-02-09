// Please make sure the following five std headers are included before hnswlib.h
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>

#include "hnswlib/hnswlib.h"

#include <AnnResultWriter.hpp>
#include <Exception.h>
#include <StringUtils.hpp>
#include <Timer.hpp>

using namespace StringUtils;
using namespace std;
using namespace hnswlib;

const size_t MAX_MEM = 1e10; // 10 GB

// Error message
#define NPP_ERROR_MSG(M)                                                       \
  do {                                                                         \
    fprintf(stderr, "%s:%d: " M, __FILE__, __LINE__);                          \
  } while (false)

// print parameters to stdout
void show_params(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  while (*fmt != '\0') {
    char *name = va_arg(args, char *);
    if (*fmt == 'i') {
      int val = va_arg(args, int);
      printf("%s: %d\n", name, val);
    } else if (*fmt == 'c') {
      int val = va_arg(args, int);
      printf("%s: \'%c\'\n", name, val);
    } else if (*fmt == 'f') {
      double val = va_arg(args, double);
      printf("%s: %f\n", name, val);
    } else if (*fmt == 's') {
      char *val = va_arg(args, char *);
      printf("%s: \"%s\"\n", name, val);
    } else {
      NPP_ERROR_MSG("Unsupported format");
    }
    ++fmt;
  }

  va_end(args);
}

void usage() {
  printf("HNSW (v1.0)\n");
  printf("Options\n");
  printf("-d {value}     \trequired \tdimensionality\n");
  printf("-ds {string}   \trequired \tdataset file\n");
  printf("-n {value}     \trequired \tcardinality\n");
  printf("-M {value}     \trequired \tparameter M (maximum number of outgoing "
         "connections in the graph)\n");
  printf("-E {value}     \trequired (only for build) \tparameter "
         "efConstruction (construction time/accuracy trade-off)\n");
  printf(
      "-if {string}   \trequired \tfile path for hnsw index folder (for build) "
      "or index file (for query)\n");
  printf("-k {value}     \toptional(only for query) \tnumber of neighbors "
         "(default: 1)\n");
  printf(
      "-gt {string}   \trequired (only for query) \tfile of exact results\n");
  printf("-rf {string}   \trequired (only for query) \tresult file\n");
  printf("-qs {string}   \trequired (only for query) \tfile of query set\n");

  printf("\n");
  printf("Build a hnsw index\n");
  printf("-d -n -ds -n -M -E -if\n");
  printf("Use hnsw to answer a knn workload\n");
  printf("-d -n -ds -n -if -rf -qs -gt [-k]\n");

  printf("\n");
}

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

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  /* AIX and Solaris ------------------------------------------ */
  struct psinfo psinfo;
  int fd = -1;
  if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
    return (size_t)0L; /* Can't open? */
  if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
    close(fd);
    return (size_t)0L; /* Can't read? */
  }
  close(fd);
  return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))
  /* BSD, Linux, and OSX -------------------------------------- */
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
  return (size_t)rusage.ru_maxrss;
#else
  return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
  /* Unknown OS ----------------------------------------------- */
  return (size_t)0L; /* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &infoCount) != KERN_SUCCESS)
    return (size_t)0L; /* Can't access? */
  return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)
  /* Linux ---------------------------------------------------- */
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t)0L; /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t)0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
  /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
  return (size_t)0L; /* Unsupported. */
#endif
}

void indexing(const char *ds_filename, // database filename
              const char *i_filename,  // index filename (output)
              unsigned num,            // # of points
              unsigned dim,            // dimension
              unsigned M,              // parameter M
              unsigned efConstruction  // parameter efConstruction
) {
  std::vector<float> train;
  load_data(ds_filename, train, num, dim);

  L2Space l2space(dim);

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


  AnnResultWriter writer(perf_filename);
  writer.writeRow("s", "dsname,#n,#dim,M,efConstruction,index_size(bytes),"
                       "construction_time(us),Peak_Mem(bytes)");
  const char *fmt = "siiiiifi";

  HighResolutionTimer timer;
  timer.restart();
  appr_alg = new HierarchicalNSW<float>(&l2space, num, M, efConstruction);
  for (auto i = 0; i < num; ++i) {
    appr_alg->addPoint((void *)(&train[i * dim]), (size_t)i);
  }
  auto e = timer.elapsed();
  auto peak_mem = getPeakRSS();

  appr_alg->saveIndex(i_filename);

  size_t isz = getCurrentRSS();

  writer.writeRow(fmt, ds_filename, num, dim, M, efConstruction, isz, e,peak_mem);
}

void knn(const char *ds_filename, // database filename
         const char *q_filename,  // query filename
         const char *i_filename,  // index filename
         const char *gt_filename, // ground truth filename
         const char *r_filename,  // result filename
         unsigned num,            // # of points in database
         unsigned dim,            // dimensionality
         unsigned qn,             // # of queries
         unsigned K               // # of NNs
) {
  std::vector<float> train, test;
  load_data(ds_filename, train, num, dim);

  load_data(q_filename, test, qn, dim);

  L2Space l2space(dim);

  HierarchicalNSW<float> *appr_alg = nullptr;
  appr_alg = new HierarchicalNSW<float>(&l2space, i_filename, false);

  unsigned r_qn, r_maxk;
  FILE *fp = fopen(gt_filename, "r");
  NPP_ENFORCE(fp != NULL);
  NPP_ENFORCE(fscanf(fp, "%d %d\n", &r_qn, &r_maxk) >= 0);
  NPP_ENFORCE(r_qn >= qn && r_maxk >= K);

  std::vector<float> gt(qn * r_maxk, -1.0f);

  for (unsigned i = 0; i < qn; ++i) {
    unsigned j;
    NPP_ENFORCE(fscanf(fp, "%d", &j) >= 0);
    NPP_ENFORCE(j == i);
    for (j = 0; j < r_maxk; ++j) {
      NPP_ENFORCE(fscanf(fp, " %f", &gt[i * r_maxk + j]) >= 0);
    }
    NPP_ENFORCE(fscanf(fp, "\n") >= 0);
  }
  NPP_ENFORCE(fclose(fp) == 0);

  HighResolutionTimer timer;
  AnnResultWriter writer(r_filename);

  writer.writeRow("s", AnnResults::_DEFAULT_HEADER_I_);

  for (unsigned i = 0; i < qn; i++) {
    timer.restart();
    std::priority_queue<std::pair<float, labeltype>> result =
        appr_alg->searchKnn(&test[dim * i], K);
    auto query_time = timer.elapsed();

    for (unsigned j = 0; j < K; ++j) {
      auto res = result.top();
      result.pop();
      float dist = std::round(res.first);

      int gdist = std::round(gt[i * r_maxk + j] * gt[i * r_maxk + j]);
      writer.writeRow(AnnResults::_DEFAULT_FMT_I_, i, j, res.second, (int)dist,
                      gdist, (dist / gdist), query_time);
    }
  }
}

/*
 * Get the index of next unblank char from a string.
 */
int GetNextChar(char *str) {
  int rtn = 0;

  // Jump over all blanks
  while (str[rtn] == ' ') {
    rtn++;
  }

  return rtn;
}

/*
 * Get next word from a string.
 */
void GetNextWord(char *str, char *word) {
  // Jump over all blanks
  while (*str == ' ') {
    str++;
  }

  while (*str != ' ' && *str != '\0') {
    *word = *str;
    str++;
    word++;
  }

  *word = '\0';
}

int main(int argc, char **argv) {
  // These two are global variables
  unsigned nPoints = 0;         // the number of points
  unsigned pointsDimension = 0; // the dimensionality of points

  int qn = -1; // the number of queries
  int k = 1;   // the k of k-NN

  char ds[200] = "";  // the file path of dataset
  char qs[200] = "";  // the file path of query set
  char gt[200] = "";  // the file path of ground truth
  char rf[200] = "";  // the folder path of results
  char inf[200] = ""; // the folder path of index

  int M = -1;
  int efConstruction = -1;

  int cnt = 1;
  bool failed = false;
  char *arg;
  int i;
  char para[10];

  std::string err_msg;

  while (cnt < argc && !failed) {
    arg = argv[cnt++];
    if (cnt == argc) {
      failed = true;
      break;
    }

    i = GetNextChar(arg);
    if (arg[i] != '-') {
      failed = true;
      err_msg = "Wrong format!";
      break;
    }

    GetNextWord(arg + i + 1, para);

    arg = argv[cnt++];

    if (strcmp(para, "n") == 0) {
      nPoints = atoi(arg);
      if (nPoints <= 0) {
        failed = true;
        err_msg = "n should a positive integer!";
        break;
      }
    } else if (strcmp(para, "d") == 0) {
      pointsDimension = atoi(arg);
      if (pointsDimension <= 0) {
        failed = true;
        err_msg = "d should a positive integer!";
        break;
      }
    } else if (strcmp(para, "qn") == 0) {
      qn = atoi(arg);
      if (qn <= 0) {
        failed = true;
        err_msg = "qn should a positive integer!";
        break;
      }
    } else if (strcmp(para, "k") == 0) {
      k = atoi(arg);
      if (k <= 0) {
        failed = true;
        err_msg = "k should a positive integer!";
        break;
      }
    } else if (strcmp(para, "M") == 0) {
      M = atoi(arg);
      if (M <= 0) {
        failed = true;
        err_msg = "M should a positive integer!";
        break;
      }
    } else if (strcmp(para, "E") == 0) {
      efConstruction = atoi(arg);
      if (efConstruction <= 0) {
        failed = true;
        err_msg = "E should a positive integer!";
        break;
      }
    } else if (strcmp(para, "ds") == 0) {
      GetNextWord(arg, ds);

    } else if (strcmp(para, "qs") == 0) {
      GetNextWord(arg, qs);

    } else if (strcmp(para, "gt") == 0) {
      GetNextWord(arg, gt);

    } else if (strcmp(para, "if") == 0) {
      GetNextWord(arg, inf);

    } else if (strcmp(para, "rf") == 0) {
      GetNextWord(arg, rf);

    } else {
      failed = true;
      fprintf(stderr, "Unknown option -%s!\n\n", para);
    }
  }

  if (failed) {
    fprintf(stderr, "%s:%d: %s\n\n", __FILE__, __LINE__, err_msg.c_str());
    usage();
    return EXIT_FAILURE;
  }

  int nargs = (cnt - 1) / 2;

  if (!(nargs == 6 || nargs == 9 || nargs == 8)) {
    fprintf(stderr, "%s:%d: %s\n\n", __FILE__, __LINE__,
            "Wrong number of arguements!");
    usage();
    return EXIT_FAILURE;
  }

#ifndef DISABLE_VERBOSE
  printf("=====================================================\n");
  show_params("iiiiiissss", "# of points", nPoints, "dimension",
              pointsDimension, "# of queries", qn, "M", M, "efConstruction",
              efConstruction, "k", k, "dataset filename", ds, "index folder",
              inf, "result filename", rf, "ground truth filename", gt);
  printf("=====================================================\n");
#endif

  try {
    if (nargs == 6) {
      NPP_ENFORCE(strlen(ds) != 0);

      NPP_ENFORCE(strlen(inf) != 0);
      NPP_ENFORCE(M > 0 && efConstruction > 0 && nPoints > 0 &&
                  pointsDimension > 0);
      indexing(ds, inf, nPoints, pointsDimension, M, efConstruction);
    } else {
      NPP_ENFORCE(strlen(ds) != 0);
      NPP_ENFORCE(strlen(qs) != 0);
      NPP_ENFORCE(strlen(gt) != 0);
      NPP_ENFORCE(strlen(inf) != 0);
      NPP_ENFORCE(strlen(rf) != 0);
      NPP_ENFORCE(nPoints > 0 && pointsDimension > 0 && qn > 0 && k > 0);
      knn(ds, qs, inf, gt, rf, nPoints, pointsDimension, qn, k);
    }
  } catch (const npp::Exception &e) {
    std::cerr << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}
