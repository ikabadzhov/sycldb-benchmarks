#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include "../utils/sycl_device.hpp"

/**
 * SSB Q1.1 Standalone Benchmark (Hardcoded Fused Kernel) - Codex0 Strategy Replication
 * 
 * This version replicates the internal logic of Codex0's modular kernels (Selection, Projection, Aggregation)
 * but fuses them into a single parallel pass over the data.
 */

// --- Codex0 Kernel Helpers (Reproduced from codex0/kernels/) ---
enum comp_op { EQ, NE, LT, LE, GT, GE };
enum logical_op { NONE, AND, OR };
enum class BinaryOp : uint8_t { Multiply, Divide, Add, Subtract };

template <typename T>
inline bool compare(comp_op CO, T a, T b) {
    switch (CO) {
        case EQ: return a == b;
        case NE: return a != b;
        case LT: return a < b;
        case LE: return a <= b;
        case GT: return a > b;
        case GE: return a >= b;
        default: return false;
    }
}

inline bool logical(logical_op logic, bool a, bool b) {
    switch (logic) {
        case AND: return a && b;
        case OR:  return a || b;
        case NONE: return b;
        default: return false;
    }
}

template <typename T>
inline T element_operation(T a, T b, BinaryOp op) {
    switch (op) {
        case BinaryOp::Multiply: return a * b;
        case BinaryOp::Divide:   return a / b;
        case BinaryOp::Add:      return a + b;
        case BinaryOp::Subtract: return a - b;
        default: return 0;
    }
}
// --- End Codex0 Helpers ---

// --- Many-Kernels Selection ---
struct Q11SelectionKernel {
    const int* lo_orderdate; const int* lo_discount; const int* lo_quantity; bool* mask;
    void operator()(sycl::id<1> idx) const {
        auto i = idx[0];
        bool f = true;
        f &= (lo_orderdate[i] >= 19930101 && lo_orderdate[i] <= 19931231);
        f &= (lo_discount[i] >= 1 && lo_discount[i] <= 3);
        f &= (lo_quantity[i] < 25);
        mask[i] = f;
    }
};

// --- Many-Kernels Aggregation ---
struct Q11AggregationKernel {
    const int* lo_discount; const int* lo_extendedprice; const bool* mask; uint64_t* result;
    void operator()(sycl::id<1> idx) const {
        auto i = idx[0];
        if (mask[i]) {
            uint64_t val = (uint64_t)lo_extendedprice[i] * (uint64_t)lo_discount[i];
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(*result);
            sum_obj.fetch_add(val);
        }
    }
};

template<typename T>
void load_column(const std::string& path, T* ptr, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Failed to open " << path << std::endl; exit(1); }
    f.read(reinterpret_cast<char*>(ptr), n * sizeof(T));
}

size_t get_file_rows(const std::string& path, size_t elem_size = 4) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return (size_t)f.tellg() / elem_size;
}

int main(int argc, char** argv) {
    int repetitions = 3;
    std::string ssb_path = "/media/ssb/sf100_columnar";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) repetitions = std::stoi(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) ssb_path = argv[++i];
    }

    sycl::queue q = sycldb::make_queue_from_args(argc, argv);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::string path_date = ssb_path + "/LINEORDER5";
    std::string path_disc = ssb_path + "/LINEORDER11";
    std::string path_quant = ssb_path + "/LINEORDER8";
    std::string path_price = ssb_path + "/LINEORDER9";

    size_t n = get_file_rows(path_date);
    if (n == 0 && ssb_path == "/media/ssb/sf100_columnar") {
        ssb_path = "/media/ssb/s100_columnar";
        n = get_file_rows(ssb_path + "/LINEORDER5");
    }

    if (n == 0) { std::cerr << "Could not find ssb files in " << ssb_path << std::endl; return 1; }
    std::cout << "Table [LINEORDER] size: " << n << " rows" << std::endl;

    int* h_date = (int*)malloc(n * sizeof(int)), *h_disc = (int*)malloc(n * sizeof(int));
    int* h_quant = (int*)malloc(n * sizeof(int)), *h_price = (int*)malloc(n * sizeof(int));

    load_column(ssb_path + "/LINEORDER5", h_date, n);
    load_column(ssb_path + "/LINEORDER11", h_disc, n);
    load_column(ssb_path + "/LINEORDER8", h_quant, n);
    load_column(ssb_path + "/LINEORDER9", h_price, n);

    int* d_date = sycl::malloc_device<int>(n, q), *d_disc = sycl::malloc_device<int>(n, q);
    int* d_quant = sycl::malloc_device<int>(n, q), *d_price = sycl::malloc_device<int>(n, q);
    uint64_t* d_res = sycl::malloc_device<uint64_t>(1, q);

    q.memcpy(d_date, h_date, n * sizeof(int));
    q.memcpy(d_disc, h_disc, n * sizeof(int));
    q.memcpy(d_quant, h_quant, n * sizeof(int));
    q.memcpy(d_price, h_price, n * sizeof(int)).wait();

    free(h_date); free(h_disc); free(h_quant); free(h_price);

    bool* d_mask = sycl::malloc_device<bool>(n, q);
    auto run_kernel = [&]() {
        q.parallel_for(sycl::range<1>(n), Q11SelectionKernel{d_date, d_disc, d_quant, d_mask});
        q.parallel_for(sycl::range<1>(n), Q11AggregationKernel{d_disc, d_price, d_mask, d_res}).wait();
    };

    q.fill(d_res, (uint64_t)0, 1).wait();
    run_kernel();

    std::vector<double> times;
    for(int i=0; i<repetitions; ++i) {
        q.fill(d_res, (uint64_t)0, 1).wait();
        auto start = std::chrono::high_resolution_clock::now();
        run_kernel();
        auto end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(t);
        std::cout << "Iteration " << i << ": " << t << " ms" << std::endl;
    }

    double total_time = 0; for(auto t : times) total_time += t;
    double avg = total_time / times.size();
    double variance = 0; for(auto t : times) variance += (t - avg) * (t - avg);
    double stddev = std::sqrt(variance / times.size());

    uint64_t final_res = 0; q.memcpy(&final_res, d_res, 8).wait();
    std::cout << "Execution time over " << repetitions << " repetitions - Avg: " << avg << " ms, StdDev: " << stddev << " ms" << std::endl;
    std::cout << "Final result: " << final_res << std::endl;

    sycl::free(d_date, q); sycl::free(d_disc, q); sycl::free(d_quant, q); sycl::free(d_price, q); sycl::free(d_res, q); sycl::free(d_mask, q);
    return 0;
}
