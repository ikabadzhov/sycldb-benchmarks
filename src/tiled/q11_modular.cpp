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
 * SSB Q1.1 Standalone Benchmark (SYCL Tiled / Blocked execution)
 * Expands upon q11_sycldb.cpp by explicitly looping within an nd_item
 * tile (a block) using a layout similar to Mordred (ITEMS_PER_THREAD * BLOCK_THREADS).
 */

enum comp_op { EQ, NE, LT, LE, GT, GE };
enum logical_op { NONE, AND, OR };
enum class BinaryOp : uint8_t { Multiply, Divide, Add, Subtract };

template <typename T>
inline bool compare(comp_op CO, T a, T b) {
    switch (CO) {
        case EQ: return a == b; case NE: return a != b;
        case LT: return a < b; case LE: return a <= b;
        case GT: return a > b; case GE: return a >= b;
        default: return false;
    }
}

inline bool logical(logical_op logic, bool a, bool b) {
    switch (logic) {
        case AND: return a && b; case OR:  return a || b;
        case NONE: return b; default: return false;
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

// --- Many-Kernels Selection (Tiled) ---
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
struct Q11SelectionTiledKernel {
    const int *lo_orderdate; const int *lo_discount; const int *lo_quantity; bool *mask; size_t num_items;
    void operator()(sycl::nd_item<1> item) const {
        int tile_offset = item.get_group(0) * (BLOCK_THREADS * ITEMS_PER_THREAD);
        int num_tile_items = (BLOCK_THREADS * ITEMS_PER_THREAD);
        if (tile_offset + num_tile_items > num_items) num_tile_items = num_items - tile_offset;
        #pragma unroll
        for(int i=0; i < ITEMS_PER_THREAD; ++i) {
            int idx = tile_offset + i * BLOCK_THREADS + item.get_local_id(0);
            if (idx < tile_offset + num_tile_items) {
                mask[idx] = (lo_orderdate[idx] >= 19930101 && lo_orderdate[idx] <= 19931231 && 
                            lo_discount[idx] >= 1 && lo_discount[idx] <= 3 && lo_quantity[idx] < 25);
            }
        }
    }
};

// --- Many-Kernels Aggregation (Tiled) ---
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
struct Q11AggregationTiledKernel {
    const int *lo_discount; const int *lo_extendedprice; const bool *mask; uint64_t *agg_result; size_t num_items;
    void operator()(sycl::nd_item<1> item) const {
        int tile_offset = item.get_group(0) * (BLOCK_THREADS * ITEMS_PER_THREAD);
        int num_tile_items = (BLOCK_THREADS * ITEMS_PER_THREAD);
        if (tile_offset + num_tile_items > num_items) num_tile_items = num_items - tile_offset;
        uint64_t sum = 0;
        #pragma unroll
        for(int i=0; i < ITEMS_PER_THREAD; ++i) {
            int idx = tile_offset + i * BLOCK_THREADS + item.get_local_id(0);
            if (idx < tile_offset + num_tile_items && mask[idx]) {
                sum += (uint64_t)lo_extendedprice[idx] * (uint64_t)lo_discount[idx];
            }
        }
        uint64_t aggregate = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<uint64_t>());
        if (item.get_local_id(0) == 0 && aggregate > 0) {
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(*agg_result);
            sum_obj.fetch_add(aggregate);
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

    size_t n = get_file_rows(ssb_path + "/LINEORDER5");
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

    const int BLOCK_THREADS = 128;
    const int ITEMS_PER_THREAD = 4;
    size_t num_blocks = (n + (BLOCK_THREADS * ITEMS_PER_THREAD) - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);
    sycl::nd_range<1> tile_range(
        sycl::range<1>(num_blocks * BLOCK_THREADS),
        sycl::range<1>(BLOCK_THREADS)
    );

    bool* d_mask = sycl::malloc_device<bool>(n, q);
    auto run_kernel = [&]() {
        q.parallel_for(tile_range, Q11SelectionTiledKernel<BLOCK_THREADS, ITEMS_PER_THREAD>{d_date, d_disc, d_quant, d_mask, n});
        q.parallel_for(tile_range, Q11AggregationTiledKernel<BLOCK_THREADS, ITEMS_PER_THREAD>{d_disc, d_price, d_mask, d_res, n}).wait();
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
