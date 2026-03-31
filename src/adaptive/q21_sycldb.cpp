#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <hipSYCL/sycl/jit.hpp>

namespace acpp_jit = sycl::AdaptiveCpp_jit;

struct Q21Context {
    const int* lo_date;
    const int* lo_part;
    const int* lo_supp;
    const int* lo_rev;
    const bool* p_filter;
    const bool* s_filter;
    const int* d_year_map;
    const int* p_brand;
    uint64_t* res_agg;
    unsigned* res_flags;
    int* res_year;
    int* res_brand;
};

// Placeholder function to be replaced at JIT time
void execute_q21_ops(sycl::item<1> idx, Q21Context ctx, bool& pass);

extern "C" {
    // These will be the implementations mapped at JIT time
    SYCL_EXTERNAL void q21_join(sycl::item<1> idx, Q21Context ctx, bool& pass) {
        size_t i = idx.get_id(0);
        pass = true;
        if (pass) pass = ctx.p_filter[ctx.lo_part[i]];
        if (pass) pass = ctx.s_filter[ctx.lo_supp[i]];
        if (pass) pass = (ctx.lo_date[i] >= 19920101 && ctx.lo_date[i] <= 19981231);
    }

    SYCL_EXTERNAL void q21_agg(sycl::item<1> idx, Q21Context ctx, bool& pass) {
        if (pass) {
            size_t i = idx.get_id(0);
            int date = ctx.lo_date[i];
            int part = ctx.lo_part[i];
            int rev = ctx.lo_rev[i];
            int year = ctx.d_year_map[date - 19920101];
            int brand = ctx.p_brand[part];
            int bucket = (year - 1992) * 100 + (brand % 100);

            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> flag_obj(ctx.res_flags[bucket]);
            if (flag_obj.exchange(1) == 0) {
                ctx.res_year[bucket] = year;
                ctx.res_brand[bucket] = brand;
            }
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(ctx.res_agg[bucket]).fetch_add((uint64_t)rev);
        }
    }
}

template<typename T>
void load_column(const std::string& path, T* ptr, size_t n) {
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(ptr), n * sizeof(T));
}

int main(int argc, char** argv) {
    std::string ssb_path = "/media/ssb/s100_columnar";
    sycl::queue q{sycl::default_selector_v};
    size_t n_fact = 600043265, n_part = 1400000, n_supp = 200000;

    int *h_tmp = (int*)malloc(n_fact * 4);
    int *d_lo_date = sycl::malloc_device<int>(n_fact, q);
    int *d_lo_part = sycl::malloc_device<int>(n_fact, q);
    int *d_lo_supp = sycl::malloc_device<int>(n_fact, q);
    int *d_lo_rev = sycl::malloc_device<int>(n_fact, q);

    load_column(ssb_path + "/LINEORDER5", h_tmp, n_fact); q.memcpy(d_lo_date, h_tmp, n_fact * 4);
    load_column(ssb_path + "/LINEORDER3", h_tmp, n_fact); q.memcpy(d_lo_part, h_tmp, n_fact * 4);
    load_column(ssb_path + "/LINEORDER4", h_tmp, n_fact); q.memcpy(d_lo_supp, h_tmp, n_fact * 4);
    load_column(ssb_path + "/LINEORDER12", h_tmp, n_fact); q.memcpy(d_lo_rev, h_tmp, n_fact * 4).wait();
    free(h_tmp);

    bool *d_p_filter = sycl::malloc_device<bool>(n_part + 1, q);
    bool *d_s_filter = sycl::malloc_device<bool>(n_supp + 1, q);
    int *d_year_map = sycl::malloc_device<int>(3000, q);
    int *d_p_brand = sycl::malloc_device<int>(n_part + 1, q);
    q.fill(d_p_filter, true, n_part + 1);
    q.fill(d_s_filter, true, n_supp + 1);
    q.fill(d_year_map, 1992, 3000);
    q.fill(d_p_brand, 1, n_part + 1).wait();

    uint64_t *d_res_agg = sycl::malloc_device<uint64_t>(1000, q);
    unsigned *d_res_flags = sycl::malloc_device<unsigned>(1000, q);
    int *d_res_year = sycl::malloc_device<int>(1000, q);
    int *d_res_brand = sycl::malloc_device<int>(1000, q);

    Q21Context ctx{d_lo_date, d_lo_part, d_lo_supp, d_lo_rev, d_p_filter, d_s_filter, d_year_map, d_p_brand, d_res_agg, d_res_flags, d_res_year, d_res_brand};

    acpp_jit::dynamic_function_config cfg;
    // Requests calls to execute_q21_ops to be replaced at JIT time with {q21_join(); q21_agg();}
    cfg.define_as_call_sequence(&execute_q21_ops, {&q21_join, &q21_agg});

    auto run_kernel = [&]() {
        q.parallel_for(sycl::range<1>{n_fact}, cfg.apply([=](sycl::item<1> idx) {
            bool pass = true;
            execute_q21_ops(idx, ctx, pass);
        })).wait();
    };

    q.fill(d_res_agg, 0ULL, 1000);
    q.fill(d_res_flags, 0u, 1000).wait();
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
