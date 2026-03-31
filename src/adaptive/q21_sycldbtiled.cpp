#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <hipSYCL/sycl/jit.hpp>

namespace acpp_jit = sycl::AdaptiveCpp_jit;

struct Q21Context {
    const int* d_lo_date;
    const int* d_lo_part;
    const int* d_lo_supp;
    const int* d_lo_rev;
    const bool* pf;
    const bool* sf;
    const int* dym;
    const int* pb;
    uint64_t* res;
    unsigned* flags;
    int* ry;
    int* rb;
};

// Placeholder for JIT fusion
void execute_q21_tiled_ops(sycl::nd_item<1> it, Q21Context ctx, bool& pass, int idx);

extern "C" {
    SYCL_EXTERNAL void q21_probe_t(sycl::nd_item<1> it, Q21Context ctx, bool& pass, int idx) {
        pass = (ctx.pf[ctx.d_lo_part[idx]] && ctx.sf[ctx.d_lo_supp[idx]] && ctx.d_lo_date[idx] >= 19920101 && ctx.d_lo_date[idx] <= 19981231);
    }

    SYCL_EXTERNAL void q21_agg_t(sycl::nd_item<1> it, Q21Context ctx, bool& pass, int idx) {
        if(pass) {
            int d = ctx.d_lo_date[idx]; int p = ctx.d_lo_part[idx]; int r = ctx.d_lo_rev[idx];
            int year = ctx.dym[d - 19920101]; int brand = ctx.pb[p]; int bucket = (year - 1992) * 100 + (brand % 100);
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> flag_obj(ctx.flags[bucket]);
            if (flag_obj.exchange(1) == 0) { ctx.ry[bucket] = year; ctx.rb[bucket] = brand; }
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(ctx.res[bucket]).fetch_add((uint64_t)r);
        }
    }
}

template<typename T>
void load_column(const std::string& path, T* ptr, size_t n) {
    std::ifstream f(path, std::ios::binary); f.read(reinterpret_cast<char*>(ptr), n * sizeof(T));
}

int main(int argc, char** argv) {
    std::string ssb_path = "/media/ssb/s100_columnar"; sycl::queue q{sycl::default_selector_v};
    size_t n = 600043265, n_part = 1400000, n_supp = 200000;
    int *h_tmp = (int*)malloc(n*4);
    int *d_lo_date = sycl::malloc_device<int>(n, q), *d_lo_part = sycl::malloc_device<int>(n, q), *d_lo_supp = sycl::malloc_device<int>(n, q), *d_lo_rev = sycl::malloc_device<int>(n, q);
    load_column(ssb_path + "/LINEORDER5", h_tmp, n); q.memcpy(d_lo_date, h_tmp, n*4);
    load_column(ssb_path + "/LINEORDER3", h_tmp, n); q.memcpy(d_lo_part, h_tmp, n*4);
    load_column(ssb_path + "/LINEORDER4", h_tmp, n); q.memcpy(d_lo_supp, h_tmp, n*4);
    load_column(ssb_path + "/LINEORDER12", h_tmp, n); q.memcpy(d_lo_rev, h_tmp, n*4).wait();
    free(h_tmp);

    bool *d_p_filter = sycl::malloc_device<bool>(n_part+1, q), *d_s_filter = sycl::malloc_device<bool>(n_supp+1, q);
    int *d_year_map = sycl::malloc_device<int>(3000, q), *d_p_brand = sycl::malloc_device<int>(n_part+1, q);
    q.fill(d_p_filter, true, n_part+1); q.fill(d_s_filter, true, n_supp+1); q.fill(d_year_map, 1992, 3000); q.fill(d_p_brand, 1, n_part+1).wait();

    uint64_t *d_res_agg = sycl::malloc_device<uint64_t>(1000, q); unsigned *d_res_flags = sycl::malloc_device<unsigned>(1000, q);
    int *d_res_year = sycl::malloc_device<int>(1000, q), *d_res_brand = sycl::malloc_device<int>(1000, q);

    Q21Context ctx{d_lo_date, d_lo_part, d_lo_supp, d_lo_rev, d_p_filter, d_s_filter, d_year_map, d_p_brand, d_res_agg, d_res_flags, d_res_year, d_res_brand};

    acpp_jit::dynamic_function_config cfg;
    cfg.define_as_call_sequence(&execute_q21_tiled_ops, {&q21_probe_t, &q21_agg_t});

    const int BLOCK_SIZE = 128; const int ITEMS = 4;
    size_t num_groups = (n + BLOCK_SIZE * ITEMS - 1) / (BLOCK_SIZE * ITEMS);
    sycl::nd_range<1> ndr{num_groups * BLOCK_SIZE, BLOCK_SIZE};

    auto run_kernel = [&]() {
        q.parallel_for(ndr, cfg.apply([=](sycl::nd_item<1> it) {
            int gid = it.get_group(0); int lid = it.get_local_id(0);
            int base = gid * BLOCK_SIZE * ITEMS + lid;
            #pragma unroll
            for(int i=0; i<ITEMS; ++i) {
                int idx = base + i * BLOCK_SIZE;
                if(idx < n) {
                    bool pass = true;
                    execute_q21_tiled_ops(it, ctx, pass, idx);
                }
            }
        })).wait();
    };

    q.fill(d_res_agg, 0ULL, 1000); q.fill(d_res_flags, 0u, 1000).wait();
    run_kernel(); // Warmup and JIT trigger

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<10; ++i) {
        q.fill(d_res_agg, 0ULL, 1000);
        q.fill(d_res_flags, 0u, 1000).wait();
        run_kernel();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Avg: " << std::chrono::duration<double, std::milli>(end - start).count() / 10.0 << " ms" << std::endl;
    return 0;
}
