#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

/**
 * SSB Q1.1 Standalone Benchmark (Hardcoded CUDA - Mordred Replica)
 * Replicates the block-striped memory load, predicate evaluation, and timing strategy 
 * found in the original Mordred GPU codebase.
 */

inline void cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

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

// Mordred style kernel logic. 128 threads, 4 items each, loaded striped (coalesced).
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q11_mordred_kernel(const int* __restrict__ lo_orderdate,
                                   const int* __restrict__ lo_discount,
                                   const int* __restrict__ lo_quantity,
                                   const int* __restrict__ lo_extendedprice,
                                   int num_items,
                                   unsigned long long* __restrict__ revenue) {
                                       
    int items_orderdate[ITEMS_PER_THREAD];
    int items_discount[ITEMS_PER_THREAD];
    int items_quantity[ITEMS_PER_THREAD];
    int items_extendedprice[ITEMS_PER_THREAD];
    bool selection_flags[ITEMS_PER_THREAD];

    long long sum = 0;
    int tile_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int num_tile_items = (BLOCK_THREADS * ITEMS_PER_THREAD);
    if (tile_offset + num_tile_items > num_items) {
        num_tile_items = num_items - tile_offset;
    }

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) selection_flags[i] = true;

    // Load discount & evaluate
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items) {
            items_discount[i] = lo_discount[idx];
            selection_flags[i] = selection_flags[i] && (items_discount[i] >= 1 && items_discount[i] <= 3);
        } else {
            selection_flags[i] = false;
        }
    }

    // Load quantity & evaluate
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items && selection_flags[i]) {
            items_quantity[i] = lo_quantity[idx];
            selection_flags[i] = selection_flags[i] && (items_quantity[i] < 25);
        }
    }

    // Load orderdate & evaluate
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items && selection_flags[i]) {
            items_orderdate[i] = lo_orderdate[idx];
            selection_flags[i] = selection_flags[i] && (items_orderdate[i] >= 19930101 && items_orderdate[i] <= 19931231);
        }
    }

    // Load extendedprice & calculate sum
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items && selection_flags[i]) {
            items_extendedprice[i] = lo_extendedprice[idx];
            sum += (long long)items_extendedprice[i] * items_discount[i];
        }
    }

    // BlockSum reduction
    typedef cub::BlockReduce<long long, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    long long aggregate = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(revenue, (unsigned long long)aggregate);
    }
}

int main(int argc, char** argv) {
    int repetitions = 3; std::string ssb_path = "/media/ssb/sf100_columnar";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) repetitions = std::stoi(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) ssb_path = argv[++i];
    }

    size_t n = get_file_rows(ssb_path + "/LINEORDER5");
    if (n == 0 && ssb_path == "/media/ssb/sf100_columnar") { ssb_path = "/media/ssb/s100_columnar"; n = get_file_rows(ssb_path + "/LINEORDER5"); }
    if (n == 0) return 1;

    std::cout << "Table [LINEORDER] size: " << n << " rows" << std::endl;

    int *h_date = (int*)malloc(n*4), *h_disc = (int*)malloc(n*4), *h_quant = (int*)malloc(n*4), *h_price = (int*)malloc(n*4);
    load_column(ssb_path + "/LINEORDER5", h_date, n);
    load_column(ssb_path + "/LINEORDER11", h_disc, n);
    load_column(ssb_path + "/LINEORDER8", h_quant, n);
    load_column(ssb_path + "/LINEORDER9", h_price, n);

    int *d_date, *d_disc, *d_quant, *d_price;
    CUDA_CHECK(cudaMalloc(&d_date, n*4)); CUDA_CHECK(cudaMalloc(&d_disc, n*4));
    CUDA_CHECK(cudaMalloc(&d_quant, n*4)); CUDA_CHECK(cudaMalloc(&d_price, n*4));
    unsigned long long *d_res; CUDA_CHECK(cudaMalloc(&d_res, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_date, h_date, n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_disc, h_disc, n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quant, h_quant, n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_price, h_price, n*4, cudaMemcpyHostToDevice));
    free(h_date); free(h_disc); free(h_quant); free(h_price);

    const int BLOCK_THREADS = 128;
    const int ITEMS_PER_THREAD = 4;
    int num_blocks = (n + (BLOCK_THREADS * ITEMS_PER_THREAD) - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);

    // Warmup
    CUDA_CHECK(cudaMemset(d_res, 0, sizeof(unsigned long long)));
    q11_mordred_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(d_date, d_disc, d_quant, d_price, n, d_res);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> times;
    for(int i=0; i<repetitions; ++i) {
        CUDA_CHECK(cudaMemset(d_res, 0, sizeof(unsigned long long)));
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        
        // Mordred begins timing exactly here
        cudaEventRecord(start);
        
        q11_mordred_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(d_date, d_disc, d_quant, d_price, n, d_res);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Mordred stops timing here
        float t_ms = 0;
        cudaEventElapsedTime(&t_ms, start, stop);
        times.push_back(t_ms);
        std::cout << "Iteration " << i << ": " << t_ms << " ms" << std::endl;
        
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    double avg = 0; for(auto t : times) avg += t; avg /= times.size();
    double var = 0; for(auto t : times) var += (t-avg)*(t-avg); double stddev = std::sqrt(var/times.size());

    unsigned long long final_res = 0;
    CUDA_CHECK(cudaMemcpy(&final_res, d_res, 8, cudaMemcpyDeviceToHost));
    std::cout << "Execution time over " << repetitions << " repetitions - Avg: " << avg << " ms, StdDev: " << stddev << " ms" << std::endl;
    std::cout << "Final result: " << final_res << std::endl;

    cudaFree(d_date); cudaFree(d_disc); cudaFree(d_quant); cudaFree(d_price); cudaFree(d_res);
    return 0;
}
