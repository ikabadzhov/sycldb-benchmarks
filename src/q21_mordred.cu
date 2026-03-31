#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

/**
 * SSB Q2.1 Standalone Benchmark (Hardcoded CUDA - Mordred Replica)
 * Replicates Mordred's explicit build/probe phase separation and timing.
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

// ----------------- BUILD KERNELS -----------------
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_ht_s(const int* __restrict__ s_region, const int* __restrict__ s_suppkey, int num_items, int* __restrict__ ht_s) {
    int tile_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int num_tile_items = (BLOCK_THREADS * ITEMS_PER_THREAD);
    if (tile_offset + num_tile_items > num_items) num_tile_items = num_items - tile_offset;
    
    int items_region[ITEMS_PER_THREAD];
    int items_suppkey[ITEMS_PER_THREAD];
    bool selection_flags[ITEMS_PER_THREAD];
    
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items) {
            items_region[i] = s_region[idx];
            selection_flags[i] = (items_region[i] == 1);
        } else { selection_flags[i] = false; }
    }
    
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items && selection_flags[i]) {
            items_suppkey[i] = s_suppkey[idx];
            if (items_suppkey[i] < num_items + 1) {
                ht_s[items_suppkey[i]] = items_suppkey[i];
            }
        }
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_ht_p(const int* __restrict__ p_category, const int* __restrict__ p_partkey, const int* __restrict__ p_brand1, int num_items, int* __restrict__ ht_p) {
    int tile_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int num_tile_items = (BLOCK_THREADS * ITEMS_PER_THREAD);
    if (tile_offset + num_tile_items > num_items) num_tile_items = num_items - tile_offset;
    
    int items_category[ITEMS_PER_THREAD];
    int items_partkey[ITEMS_PER_THREAD];
    int items_brand[ITEMS_PER_THREAD];
    bool selection_flags[ITEMS_PER_THREAD];
    
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items) {
            items_category[i] = p_category[idx];
            selection_flags[i] = (items_category[i] == 1);
        } else { selection_flags[i] = false; }
    }
    
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items && selection_flags[i]) {
            items_partkey[i] = p_partkey[idx];
            items_brand[i] = p_brand1[idx];
            if (items_partkey[i] * 2 + 1 < num_items * 2 + 2) {
                ht_p[items_partkey[i] * 2] = items_partkey[i];
                ht_p[items_partkey[i] * 2 + 1] = items_brand[i];
            }
        }
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_ht_d(const int* __restrict__ d_datekey, const int* __restrict__ d_year, int num_items, int* __restrict__ ht_d, int val_min) {
    int tile_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int num_tile_items = (BLOCK_THREADS * ITEMS_PER_THREAD);
    if (tile_offset + num_tile_items > num_items) num_tile_items = num_items - tile_offset;
    
    int items_datekey[ITEMS_PER_THREAD];
    int items_year[ITEMS_PER_THREAD];
    
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items) {
            items_datekey[i] = d_datekey[idx];
            items_year[i] = d_year[idx];
            
            int hash = items_datekey[i] - val_min;
            if (hash >= 0 && hash * 2 + 1 < 100000) {
                ht_d[hash * 2] = items_datekey[i];
                ht_d[hash * 2 + 1] = items_year[i];
            }
        }
    }
}

// ----------------- PROBE KERNEL -----------------

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q21(const int* __restrict__ lo_orderdate, 
                          const int* __restrict__ lo_partkey, 
                          const int* __restrict__ lo_suppkey, 
                          const int* __restrict__ lo_revenue, 
                          int num_items,
                          const int* __restrict__ ht_s,
                          const int* __restrict__ ht_p,
                          const int* __restrict__ ht_d,
                          int* __restrict__ res) { 

    int tile_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int num_tile_items = (BLOCK_THREADS * ITEMS_PER_THREAD);
    if (tile_offset + num_tile_items > num_items) num_tile_items = num_items - tile_offset;

    int items_suppkey[ITEMS_PER_THREAD];
    int items_partkey[ITEMS_PER_THREAD];
    int items_orderdate[ITEMS_PER_THREAD];
    int items_revenue[ITEMS_PER_THREAD];
    
    int brands[ITEMS_PER_THREAD];
    int years[ITEMS_PER_THREAD];
    bool selection_flags[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) selection_flags[i] = true;

    // Load suppkey
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items) {
            items_suppkey[i] = lo_suppkey[idx];
            if (items_suppkey[i] >= 0 && items_suppkey[i] < 300000) {
                int match = ht_s[items_suppkey[i]];
                if (match == 0) selection_flags[i] = false;
            } else {
                selection_flags[i] = false;
            }
        } else { selection_flags[i] = false; }
    }

    // Load partkey
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items && selection_flags[i]) {
            items_partkey[i] = lo_partkey[idx];
            if (items_partkey[i] >= 0 && items_partkey[i] < 2000000) {
                int match = ht_p[items_partkey[i] * 2];
                if (match != 0) {
                    brands[i] = ht_p[items_partkey[i] * 2 + 1];
                } else {
                    selection_flags[i] = false;
                }
            } else {
                selection_flags[i] = false;
            }
        }
    }

    // Load orderdate
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items && selection_flags[i]) {
            items_orderdate[i] = lo_orderdate[idx];
            int hash = items_orderdate[i] - 19920101;
            if (hash >= 0 && hash < 50000) {
                int match = ht_d[hash * 2];
                if (match != 0) {
                    years[i] = ht_d[hash * 2 + 1];
                } else {
                    selection_flags[i] = false;
                }
            } else {
                selection_flags[i] = false;
            }
        }
    }

    // Load revenue
    #pragma unroll
    for (int i=0; i<ITEMS_PER_THREAD; ++i) {
        int idx = tile_offset + i * BLOCK_THREADS + threadIdx.x;
        if (idx < tile_offset + num_tile_items && selection_flags[i]) {
            items_revenue[i] = lo_revenue[idx];

            int hash = std::abs((brands[i] * 7 + (years[i] - 1992))) % ((1998-1992+1) * 1000);
            res[hash * 4] = years[i];
            res[hash * 4 + 1] = brands[i];
            atomicAdd((unsigned long long*)&res[hash * 4 + 2], (unsigned long long)items_revenue[i]);
        }
    }
}

int main(int argc, char** argv) {
    int repetitions = 3; std::string ssb_path = "/media/ssb/sf100_columnar";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) repetitions = std::stoi(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) ssb_path = argv[++i];
    }

    size_t n_fact = get_file_rows(ssb_path + "/LINEORDER5");
    if (n_fact == 0 && ssb_path == "/media/ssb/sf100_columnar") { ssb_path = "/media/ssb/s100_columnar"; n_fact = get_file_rows(ssb_path + "/LINEORDER5"); }
    size_t n_part = get_file_rows(ssb_path + "/PART0"), n_supp = get_file_rows(ssb_path + "/SUPPLIER0"), n_date = get_file_rows(ssb_path + "/DDATE0");
    if (n_fact == 0) return 1;

    std::cout << "Table sizes: Fact=" << n_fact << ", Part=" << n_part << ", Supp=" << n_supp << ", Date=" << n_date << std::endl;

    int *h_lo_date = (int*)malloc(n_fact * 4), *h_lo_part = (int*)malloc(n_fact * 4), *h_lo_supp = (int*)malloc(n_fact * 4), *h_lo_rev = (int*)malloc(n_fact * 4);
    load_column(ssb_path + "/LINEORDER5", h_lo_date, n_fact); load_column(ssb_path + "/LINEORDER3", h_lo_part, n_fact);
    load_column(ssb_path + "/LINEORDER4", h_lo_supp, n_fact); load_column(ssb_path + "/LINEORDER12", h_lo_rev, n_fact);

    int *h_p_key = (int*)malloc(n_part * 4), *h_p_cat = (int*)malloc(n_part * 4), *h_brand = (int*)malloc(n_part * 4);
    load_column(ssb_path + "/PART0", h_p_key, n_part); load_column(ssb_path + "/PART3", h_p_cat, n_part); load_column(ssb_path + "/PART4", h_brand, n_part);
    
    int *h_s_key = (int*)malloc(n_supp * 4), *h_s_reg = (int*)malloc(n_supp * 4);
    load_column(ssb_path + "/SUPPLIER0", h_s_key, n_supp); load_column(ssb_path + "/SUPPLIER5", h_s_reg, n_supp);
    
    int *h_d_key = (int*)malloc(n_date * 4), *h_d_year = (int*)malloc(n_date * 4);
    load_column(ssb_path + "/DDATE0", h_d_key, n_date); load_column(ssb_path + "/DDATE4", h_d_year, n_date);

    // Device allocations for Fact
    int *d_lo_date, *d_lo_part, *d_lo_supp, *d_lo_rev;
    CUDA_CHECK(cudaMalloc(&d_lo_date, n_fact*4)); CUDA_CHECK(cudaMalloc(&d_lo_part, n_fact*4));
    CUDA_CHECK(cudaMalloc(&d_lo_supp, n_fact*4)); CUDA_CHECK(cudaMalloc(&d_lo_rev, n_fact*4));
    CUDA_CHECK(cudaMemcpy(d_lo_date, h_lo_date, n_fact*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lo_part, h_lo_part, n_fact*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lo_supp, h_lo_supp, n_fact*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lo_rev, h_lo_rev, n_fact*4, cudaMemcpyHostToDevice));

    // Device allocations for Dims
    int *d_p_key, *d_p_cat, *d_p_brand;
    CUDA_CHECK(cudaMalloc(&d_p_key, n_part*4)); CUDA_CHECK(cudaMalloc(&d_p_cat, n_part*4)); CUDA_CHECK(cudaMalloc(&d_p_brand, n_part*4));
    CUDA_CHECK(cudaMemcpy(d_p_key, h_p_key, n_part*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_cat, h_p_cat, n_part*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_brand, h_brand, n_part*4, cudaMemcpyHostToDevice));
    
    int *d_s_key, *d_s_reg;
    CUDA_CHECK(cudaMalloc(&d_s_key, n_supp*4)); CUDA_CHECK(cudaMalloc(&d_s_reg, n_supp*4));
    CUDA_CHECK(cudaMemcpy(d_s_key, h_s_key, n_supp*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_reg, h_s_reg, n_supp*4, cudaMemcpyHostToDevice));

    int *d_d_key, *d_d_year;
    CUDA_CHECK(cudaMalloc(&d_d_key, n_date*4)); CUDA_CHECK(cudaMalloc(&d_d_year, n_date*4));
    CUDA_CHECK(cudaMemcpy(d_d_key, h_d_key, n_date*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year, h_d_year, n_date*4, cudaMemcpyHostToDevice));

    const int BLOCK_THREADS = 128;
    const int ITEMS_PER_THREAD = 4;
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

    int *ht_s, *ht_p, *ht_d;
    int d_val_len = 19981230 - 19920101 + 1;
    CUDA_CHECK(cudaMalloc(&ht_d, 2 * (d_val_len + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ht_p, 2 * (n_part + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ht_s, 2 * (n_supp + 1) * sizeof(int)));

    int res_size = ((1998-1992+1) * 1000) * 4; // Mordred Q21 Hash Output Array Size
    int *d_res; CUDA_CHECK(cudaMalloc(&d_res, res_size * sizeof(int)));

    // Benchmark loop
    std::vector<double> times;
    for(int i=0; i<repetitions; ++i) {
        // Reset everything
        CUDA_CHECK(cudaMemset(ht_d, 0, 2 * (d_val_len + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(ht_p, 0, 2 * (n_part + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(ht_s, 0, 2 * (n_supp + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_res, 0, res_size * sizeof(int)));
        
        // ------------- DIMENSION BUILD KERNELS -------------
        build_ht_s<BLOCK_THREADS, ITEMS_PER_THREAD><<< (n_supp + tile_items - 1) / tile_items, BLOCK_THREADS >>>(d_s_reg, d_s_key, n_supp, ht_s);
        build_ht_p<BLOCK_THREADS, ITEMS_PER_THREAD><<< (n_part + tile_items - 1) / tile_items, BLOCK_THREADS >>>(d_p_cat, d_p_key, d_p_brand, n_part, ht_p);
        build_ht_d<BLOCK_THREADS, ITEMS_PER_THREAD><<< (n_date + tile_items - 1) / tile_items, BLOCK_THREADS >>>(d_d_key, d_d_year, n_date, ht_d, 19920101);
        
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure hashes are built
        
        // ------------- START PROBE TIMER -------------
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        probe_q21<BLOCK_THREADS, ITEMS_PER_THREAD><<< (n_fact + tile_items - 1) / tile_items, BLOCK_THREADS >>>(
            d_lo_date, d_lo_part, d_lo_supp, d_lo_rev, n_fact, ht_s, ht_p, ht_d, d_res);
        
        // ------------- STOP TIMER -------------
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float t_ms = 0;
        cudaEventElapsedTime(&t_ms, start, stop);
        times.push_back(t_ms);
        std::cout << "Iteration " << i << ": " << t_ms << " ms" << std::endl;
        
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    double avg = 0; for(auto t : times) avg += t; avg /= times.size();
    double var = 0; for(auto t : times) var += (t-avg)*(t-avg); double stddev = std::sqrt(var/times.size());

    std::cout << "Execution time over " << repetitions << " repetitions - Avg: " << avg << " ms, StdDev: " << stddev << " ms" << std::endl;

    return 0;
}
