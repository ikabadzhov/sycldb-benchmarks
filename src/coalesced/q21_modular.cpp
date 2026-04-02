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
 * SSB Q2.1 Standalone Benchmark (Hardcoded SYCL Coalesced / Vector Fetch)
 * 
 * Uses sycl::int4 (128-bit vector fetches) to guarantee optimal memory 
 * transactions per thread across the Fact table.
 */

// --- Codex0 Kernel Helpers ---
enum comp_op { EQ, NE, LT, LE, GT, GE };
enum logical_op { NONE, AND, OR };

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

struct Q21JoinCoalescedKernel {
    const sycl::int4 *lo_part, *lo_supp, *lo_date; const bool *p_filter, *s_filter;
    sycl::uchar4 *mask; size_t n_part, n_supp, n_vecs;
    void operator()(sycl::id<1> idx) const {
        int i = idx[0]; if (i >= n_vecs) return;
        sycl::int4 p = lo_part[i], s = lo_supp[i], d = lo_date[i];
        sycl::uchar4 m;
        m.x() = (p.x() >= 0 && p.x() < n_part && p_filter[p.x()] && s.x() >= 0 && s.x() < n_supp && s_filter[s.x()] && d.x() >= 19920101 && d.x() <= 19981231);
        m.y() = (p.y() >= 0 && p.y() < n_part && p_filter[p.y()] && s.y() >= 0 && s.y() < n_supp && s_filter[s.y()] && d.y() >= 19920101 && d.y() <= 19981231);
        m.z() = (p.z() >= 0 && p.z() < n_part && p_filter[p.z()] && s.z() >= 0 && s.z() < n_supp && s_filter[s.z()] && d.z() >= 19920101 && d.z() <= 19981231);
        m.w() = (p.w() >= 0 && p.w() < n_part && p_filter[p.w()] && s.w() >= 0 && s.w() < n_supp && s_filter[s.w()] && d.w() >= 19920101 && d.w() <= 19981231);
        mask[i] = m;
    }
};

struct Q21AggregationCoalescedKernel {
    const sycl::int4 *lo_date, *lo_part, *lo_revenue; const int *d_year_map, *p_brand;
    const sycl::uchar4 *mask; uint64_t *agg_result; unsigned *result_flags; int *res_year, *res_brand; size_t n_vecs;
    void operator()(sycl::id<1> idx) const {
        int i = idx[0]; if (i >= n_vecs) return;
        sycl::uchar4 m = mask[i];
        if (m.x() || m.y() || m.z() || m.w()) {
            sycl::int4 date = lo_date[i], part = lo_part[i], rev = lo_revenue[i];
            bool v[4] = {(bool)m.x(), (bool)m.y(), (bool)m.z(), (bool)m.w()};
            int ds[4] = {date.x(), date.y(), date.z(), date.w()};
            int ps[4] = {part.x(), part.y(), part.z(), part.w()};
            int rs[4] = {rev.x(), rev.y(), rev.z(), rev.w()};
            #pragma unroll
            for(int j=0; j<4; ++j) {
                if (v[j]) {
                    int year = d_year_map[ds[j] - 19920101];
                    int brand = p_brand[ps[j]];
                    int bucket = (year - 1992) * 100 + (brand % 100);
                    sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> flag_obj(result_flags[bucket]);
                    if (flag_obj.exchange(1) == 0) { res_year[bucket] = year; res_brand[bucket] = brand; }
                    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> sum_obj(agg_result[bucket]);
                    sum_obj.fetch_add((uint64_t)rs[j]);
                }
            }
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
    int repetitions = 3; std::string ssb_path = "/media/ssb/sf100_columnar";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) repetitions = std::stoi(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) ssb_path = argv[++i];
    }
    sycl::queue q = sycldb::make_queue_from_args(argc, argv);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    size_t n_fact = get_file_rows(ssb_path + "/LINEORDER5");
    if (n_fact == 0 && ssb_path == "/media/ssb/sf100_columnar") {
        ssb_path = "/media/ssb/s100_columnar";
        n_fact = get_file_rows(ssb_path + "/LINEORDER5");
    }

    size_t n_part = get_file_rows(ssb_path + "/PART0"), n_supp = get_file_rows(ssb_path + "/SUPPLIER0"), n_date = get_file_rows(ssb_path + "/DDATE0");
    if (n_fact == 0 || n_part == 0) { std::cerr << "SSB Path error: " << ssb_path << std::endl; return 1; }

    // Ensure array length is cleanly divisible by 4
    size_t padded_n = ((n_fact + 3) / 4) * 4;
    std::cout << "Table sizes: Fact=" << n_fact << " (padded " << padded_n << "), Part=" << n_part << ", Supp=" << n_supp << ", Date=" << n_date << std::endl;

    int *h_lo_date = (int*)malloc(padded_n * 4), *h_lo_part = (int*)malloc(padded_n * 4), *h_lo_supp = (int*)malloc(padded_n * 4), *h_lo_rev = (int*)malloc(padded_n * 4);
    load_column(ssb_path + "/LINEORDER5", h_lo_date, n_fact); load_column(ssb_path + "/LINEORDER3", h_lo_part, n_fact);
    load_column(ssb_path + "/LINEORDER4", h_lo_supp, n_fact); load_column(ssb_path + "/LINEORDER12", h_lo_rev, n_fact);
    for(size_t i=n_fact; i<padded_n; ++i) { h_lo_date[i]=0; h_lo_part[i]=0; h_lo_supp[i]=0; h_lo_rev[i]=0; }

    int *h_p_key = (int*)malloc(n_part * 4), *h_p_cat = (int*)malloc(n_part * 4), *h_brand = (int*)malloc(n_part * 4);
    load_column(ssb_path + "/PART0", h_p_key, n_part); load_column(ssb_path + "/PART3", h_p_cat, n_part); load_column(ssb_path + "/PART4", h_brand, n_part);
    int *h_s_key = (int*)malloc(n_supp * 4), *h_s_reg = (int*)malloc(n_supp * 4);
    load_column(ssb_path + "/SUPPLIER0", h_s_key, n_supp); load_column(ssb_path + "/SUPPLIER5", h_s_reg, n_supp);
    int *h_d_key = (int*)malloc(n_date * 4), *h_d_year = (int*)malloc(n_date * 4);
    load_column(ssb_path + "/DDATE0", h_d_key, n_date); load_column(ssb_path + "/DDATE4", h_d_year, n_date);

    bool *d_p_filter = sycl::malloc_device<bool>(n_part + 1, q); int *d_p_brand = sycl::malloc_device<int>(n_part + 1, q);
    bool *d_s_filter = sycl::malloc_device<bool>(n_supp + 1, q);
    const int D_MIN = 19920101, D_RANGE = 19981231 - 19920101 + 1;
    int *d_year_map = sycl::malloc_device<int>(D_RANGE, q);
    q.fill(d_p_filter, false, n_part+1); q.fill(d_s_filter, false, n_supp+1); q.fill(d_year_map, 0, D_RANGE).wait();

    int *dt1 = sycl::malloc_device<int>(n_part, q), *dt2 = sycl::malloc_device<int>(n_part, q), *dt3 = sycl::malloc_device<int>(n_part, q);
    q.memcpy(dt1, h_p_key, n_part*4); q.memcpy(dt2, h_p_cat, n_part*4); q.memcpy(dt3, h_brand, n_part*4);
    q.parallel_for(n_part, [=](auto i){ int k=dt1[i]; if(k>=0 && k<=n_part){ d_p_filter[k]=(dt2[i]==1); d_p_brand[k]=dt3[i]; } }).wait();
    q.memcpy(dt1, h_s_key, n_supp*4); q.memcpy(dt2, h_s_reg, n_supp*4);
    q.parallel_for(n_supp, [=](auto i){ int k=dt1[i]; if(k>=0 && k<=n_supp) d_s_filter[k]=(dt2[i]==1); }).wait();
    int *dt4 = sycl::malloc_device<int>(n_date, q), *dt5 = sycl::malloc_device<int>(n_date, q);
    q.memcpy(dt4, h_d_key, n_date * 4); q.memcpy(dt5, h_d_year, n_date * 4);
    q.parallel_for(n_date, [=](auto i){ int k=dt4[i]; if(k>=D_MIN && k<=D_MIN+D_RANGE-1) d_year_map[k-D_MIN]=dt5[i]; }).wait();
    sycl::free(dt1, q); sycl::free(dt2, q); sycl::free(dt3, q); sycl::free(dt4, q); sycl::free(dt5, q);

    sycl::int4 *d_lo_date = sycl::malloc_device<sycl::int4>(padded_n / 4, q);
    sycl::int4 *d_lo_part = sycl::malloc_device<sycl::int4>(padded_n / 4, q);
    sycl::int4 *d_lo_supp = sycl::malloc_device<sycl::int4>(padded_n / 4, q);
    sycl::int4 *d_lo_rev = sycl::malloc_device<sycl::int4>(padded_n / 4, q);

    q.memcpy(d_lo_date, h_lo_date, padded_n*4); q.memcpy(d_lo_part, h_lo_part, padded_n*4);
    q.memcpy(d_lo_supp, h_lo_supp, padded_n*4); q.memcpy(d_lo_rev, h_lo_rev, padded_n*4).wait();

    free(h_lo_date); free(h_lo_part); free(h_lo_supp); free(h_lo_rev);

    const int num_buckets = 1000;
    uint64_t *d_res_agg = sycl::malloc_device<uint64_t>(num_buckets, q);
    unsigned *d_res_flags = sycl::malloc_device<unsigned>(num_buckets, q);
    int *d_res_year = sycl::malloc_device<int>(num_buckets, q), *d_res_brand = sycl::malloc_device<int>(num_buckets, q);

    size_t n_vecs = padded_n / 4;

    sycl::uchar4 *d_mask = sycl::malloc_device<sycl::uchar4>(n_vecs, q);
    auto run_kernel = [&]() {
        q.parallel_for(sycl::range<1>(n_vecs), Q21JoinCoalescedKernel{d_lo_part, d_lo_supp, d_lo_date, d_p_filter, d_s_filter, d_mask, n_part, n_supp, n_vecs});
        q.parallel_for(sycl::range<1>(n_vecs), Q21AggregationCoalescedKernel{d_lo_date, d_lo_part, d_lo_rev, d_year_map, d_p_brand, d_mask, d_res_agg, d_res_flags, d_res_year, d_res_brand, n_vecs}).wait();
    };

    q.fill(d_res_agg, (uint64_t)0, num_buckets); q.fill(d_res_flags, 0u, num_buckets).wait();
    run_kernel();

    std::vector<double> times;
    for(int i=0; i<repetitions; ++i) {
        q.fill(d_res_agg, (uint64_t)0, num_buckets); q.fill(d_res_flags, 0u, num_buckets).wait();
        auto start = std::chrono::high_resolution_clock::now();
        run_kernel();
        auto end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(t);
        std::cout << "Iteration " << i << ": " << t << " ms" << std::endl;
    }

    double avg = 0; for(auto t : times) avg += t; avg /= times.size();
    double var = 0; for(auto t : times) var += (t-avg)*(t-avg); double stddev = std::sqrt(var/times.size());
    std::vector<uint64_t> final_agg(num_buckets);
    q.memcpy(final_agg.data(), d_res_agg, num_buckets * sizeof(uint64_t)).wait();
    uint64_t final_res = 0; for(auto v : final_agg) final_res += v;
    std::cout << "Execution time over " << repetitions << " repetitions - Avg: " << avg << " ms, StdDev: " << stddev << " ms" << std::endl;
    std::cout << "Final result: " << final_res << std::endl;

    return 0;
}
