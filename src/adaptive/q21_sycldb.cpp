#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <hipSYCL/sycl/jit.hpp>
#include "../utils/sycl_device.hpp"

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

size_t get_file_rows(const std::string& path, size_t elem_size = 4) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return (size_t)f.tellg() / elem_size;
}

int main(int argc, char** argv) {
    int repetitions = 10;
    std::string ssb_path = "/media/ssb/s100_columnar";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) repetitions = std::stoi(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) ssb_path = argv[++i];
    }
    sycl::queue q = sycldb::make_queue_from_args(argc, argv);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    size_t n_fact = get_file_rows(ssb_path + "/LINEORDER5");
    size_t n_part = get_file_rows(ssb_path + "/PART0");
    size_t n_supp = get_file_rows(ssb_path + "/SUPPLIER0");
    size_t n_date = get_file_rows(ssb_path + "/DDATE0");

    int *h_lo_date = (int*)malloc(n_fact * 4), *h_lo_part = (int*)malloc(n_fact * 4), *h_lo_supp = (int*)malloc(n_fact * 4), *h_lo_rev = (int*)malloc(n_fact * 4);
    load_column(ssb_path + "/LINEORDER5", h_lo_date, n_fact); load_column(ssb_path + "/LINEORDER3", h_lo_part, n_fact);
    load_column(ssb_path + "/LINEORDER4", h_lo_supp, n_fact); load_column(ssb_path + "/LINEORDER12", h_lo_rev, n_fact);

    int *h_p_key = (int*)malloc(n_part * 4), *h_p_cat = (int*)malloc(n_part * 4), *h_brand = (int*)malloc(n_part * 4);
    load_column(ssb_path + "/PART0", h_p_key, n_part); load_column(ssb_path + "/PART3", h_p_cat, n_part); load_column(ssb_path + "/PART4", h_brand, n_part);
    int *h_s_key = (int*)malloc(n_supp * 4), *h_s_reg = (int*)malloc(n_supp * 4);
    load_column(ssb_path + "/SUPPLIER0", h_s_key, n_supp); load_column(ssb_path + "/SUPPLIER5", h_s_reg, n_supp);
    int *h_d_key = (int*)malloc(n_date * 4), *h_d_year = (int*)malloc(n_date * 4);
    load_column(ssb_path + "/DDATE0", h_d_key, n_date); load_column(ssb_path + "/DDATE4", h_d_year, n_date);

    int *d_lo_date = sycl::malloc_device<int>(n_fact, q); int *d_lo_part = sycl::malloc_device<int>(n_fact, q);
    int *d_lo_supp = sycl::malloc_device<int>(n_fact, q); int *d_lo_rev = sycl::malloc_device<int>(n_fact, q);
    q.memcpy(d_lo_date, h_lo_date, n_fact*4); q.memcpy(d_lo_part, h_lo_part, n_fact*4);
    q.memcpy(d_lo_supp, h_lo_supp, n_fact*4); q.memcpy(d_lo_rev, h_lo_rev, n_fact*4);

    bool *d_p_filter = sycl::malloc_device<bool>(n_part + 1, q); int *d_p_brand = sycl::malloc_device<int>(n_part + 1, q);
    bool *d_s_filter = sycl::malloc_device<bool>(n_supp + 1, q);
    const int D_MIN = 19920101, D_RANGE = 19981231 - 19920101 + 1;
    int *d_year_map = sycl::malloc_device<int>(D_RANGE, q);
    q.fill(d_p_filter, false, n_part + 1); q.fill(d_s_filter, false, n_supp + 1); q.fill(d_year_map, 0, D_RANGE).wait();

    int *dt1 = sycl::malloc_device<int>(n_part, q), *dt2 = sycl::malloc_device<int>(n_part, q), *dt3 = sycl::malloc_device<int>(n_part, q);
    q.memcpy(dt1, h_p_key, n_part*4); q.memcpy(dt2, h_p_cat, n_part*4); q.memcpy(dt3, h_brand, n_part*4);
    q.parallel_for(n_part, [=](auto i){ int k=dt1[i]; if(k>=0 && k<=(int)n_part){ d_p_filter[k]=(dt2[i]==1); d_p_brand[k]=dt3[i]; } }).wait();
    q.memcpy(dt1, h_s_key, n_supp*4); q.memcpy(dt2, h_s_reg, n_supp*4);
    q.parallel_for(n_supp, [=](auto i){ int k=dt1[i]; if(k>=0 && k<=(int)n_supp) d_s_filter[k]=(dt2[i]==1); }).wait();
    int *dt4 = sycl::malloc_device<int>(n_date, q), *dt5 = sycl::malloc_device<int>(n_date, q);
    q.memcpy(dt4, h_d_key, n_date * 4); q.memcpy(dt5, h_d_year, n_date * 4);
    q.parallel_for(n_date, [=](auto i){ int k=dt4[i]; if(k>=D_MIN && k<=D_MIN + D_RANGE - 1) d_year_map[k-D_MIN]=dt5[i]; }).wait();
    sycl::free(dt1, q); sycl::free(dt2, q); sycl::free(dt3, q); sycl::free(dt4, q); sycl::free(dt5, q);

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

    std::vector<double> times;
    for(int i=0; i<repetitions; ++i) {
        q.fill(d_res_agg, 0ULL, 1000);
        q.fill(d_res_flags, 0u, 1000).wait();
        auto start = std::chrono::high_resolution_clock::now();
        run_kernel();
        auto end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(t);
        std::cout << "Run " << i << ": " << t << " ms" << std::endl;
    }
    double total = 0; for(auto t : times) total += t;
    double avg = total / times.size();
    double var = 0; for(auto t : times) var += (t-avg)*(t-avg);
    double stddev = std::sqrt(var/times.size());
    std::vector<uint64_t> final_agg(1000);
    q.memcpy(final_agg.data(), d_res_agg, 1000 * sizeof(uint64_t)).wait();
    uint64_t final_res = 0; for(auto v : final_agg) final_res += v;
    std::cout << "Avg: " << avg << " ms, StdDev: " << stddev << " ms" << std::endl;
    std::cout << "Final result: " << final_res << std::endl;
    return 0;
}
