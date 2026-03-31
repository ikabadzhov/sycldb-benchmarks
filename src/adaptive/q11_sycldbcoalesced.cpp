#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <hipSYCL/sycl/jit.hpp>

namespace acpp_jit = sycl::AdaptiveCpp_jit;

struct Q11ContextVec {
    const sycl::int4* d_date;
    const sycl::int4* d_disc;
    const sycl::int4* d_quant;
    const sycl::int4* d_price;
    uint64_t* d_res;
};

// Placeholder for JIT fusion
void execute_q11_v_ops(sycl::item<1> idx, Q11ContextVec ctx, bool* pass);

extern "C" {
    SYCL_EXTERNAL void q11_filter_v(sycl::item<1> idx, Q11ContextVec ctx, bool* pass) {
        size_t i = idx.get_id(0);
        sycl::int4 date = ctx.d_date[i];
        sycl::int4 disc = ctx.d_disc[i];
        sycl::int4 quant = ctx.d_quant[i];
        
        pass[0] = (date.x() >= 19930101 && date.x() <= 19931231 && disc.x() >= 1 && disc.x() <= 3 && quant.x() < 25);
        pass[1] = (date.y() >= 19930101 && date.y() <= 19931231 && disc.y() >= 1 && disc.y() <= 3 && quant.y() < 25);
        pass[2] = (date.z() >= 19930101 && date.z() <= 19931231 && disc.z() >= 1 && disc.z() <= 3 && quant.z() < 25);
        pass[3] = (date.w() >= 19930101 && date.w() <= 19931231 && disc.w() >= 1 && disc.w() <= 3 && quant.w() < 25);
    }

    SYCL_EXTERNAL void q11_agg_v(sycl::item<1> idx, Q11ContextVec ctx, bool* pass) {
        size_t i = idx.get_id(0);
        sycl::int4 price = ctx.d_price[i];
        sycl::int4 disc = ctx.d_disc[i];
        uint64_t sum = 0;
        if(pass[0]) sum += (uint64_t)price.x() * disc.x();
        if(pass[1]) sum += (uint64_t)price.y() * disc.y();
        if(pass[2]) sum += (uint64_t)price.z() * disc.z();
        if(pass[3]) sum += (uint64_t)price.w() * disc.w();
        if(sum > 0) {
            auto ref = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*ctx.d_res);
            ref.fetch_add(sum);
        }
    }
}

template<typename T>
void load_column(const std::string& path, T* ptr, size_t n) {
    std::ifstream f(path, std::ios::binary); f.read(reinterpret_cast<char*>(ptr), n * sizeof(T));
}

int main(int argc, char** argv) {
    std::string ssb_path = "/media/ssb/s100_columnar";
    size_t n = 600043265; size_t n_vec = (n+3)/4;
    sycl::queue q{sycl::default_selector_v};
    sycl::int4 *d_date = sycl::malloc_device<sycl::int4>(n_vec, q), *d_disc = sycl::malloc_device<sycl::int4>(n_vec, q), *d_quant = sycl::malloc_device<sycl::int4>(n_vec, q), *d_price = sycl::malloc_device<sycl::int4>(n_vec, q);
    uint64_t *d_res = sycl::malloc_device<uint64_t>(1, q);
    int *h_tmp = (int*)malloc(n_vec*16); memset(h_tmp, 0, n_vec*16);
    load_column(ssb_path + "/LINEORDER5", h_tmp, n); q.memcpy(d_date, h_tmp, n_vec*16);
    load_column(ssb_path + "/LINEORDER11", h_tmp, n); q.memcpy(d_disc, h_tmp, n_vec*16);
    load_column(ssb_path + "/LINEORDER8", h_tmp, n); q.memcpy(d_quant, h_tmp, n_vec*16);
    load_column(ssb_path + "/LINEORDER9", h_tmp, n); q.memcpy(d_price, h_tmp, n_vec*16).wait();
    free(h_tmp);

    Q11ContextVec ctx{d_date, d_disc, d_quant, d_price, d_res};

    acpp_jit::dynamic_function_config cfg;
    cfg.define_as_call_sequence(&execute_q11_v_ops, {&q11_filter_v, &q11_agg_v});

    auto run_kernel = [&]() {
        q.parallel_for(sycl::range<1>{n_vec}, cfg.apply([=](sycl::item<1> idx) {
            bool pass[4] = {true, true, true, true};
            execute_q11_v_ops(idx, ctx, pass);
        })).wait();
    };

    q.fill(d_res, 0ULL, 1).wait();
    run_kernel(); // Warmup and JIT trigger

    std::vector<double> times;
    for(int i=0; i<10; ++i) {
        q.fill(d_res, 0ULL, 1).wait();
        auto start = std::chrono::high_resolution_clock::now();
        run_kernel();
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    double total = 0; for(auto t : times) total += t;
    double avg = total / 10.0;
    double var = 0; for(auto t : times) var += (t-avg)*(t-avg);
    double stddev = std::sqrt(var/10.0);
    std::cout << "Avg: " << avg << " ms, StdDev: " << stddev << " ms" << std::endl;
    return 0;
}
