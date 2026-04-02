#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>
#include <hipSYCL/sycl/jit.hpp>

namespace acpp_jit = sycl::AdaptiveCpp_jit;

struct Q11Context {
    const int* d_date;
    const int* d_disc;
    const int* d_quant;
    const int* d_price;
    uint64_t* d_res;
};

// Placeholder for JIT fusion
void execute_q11_ops(sycl::item<1> idx, Q11Context ctx, bool& pass);

extern "C" {
    SYCL_EXTERNAL void q11_filter(sycl::item<1> idx, Q11Context ctx, bool& pass) {
        size_t i = idx.get_id(0);
        pass = true;
        if(pass) pass = (ctx.d_date[i] >= 19930101 && ctx.d_date[i] <= 19931231);
        if(pass) pass = (ctx.d_disc[i] >= 1 && ctx.d_disc[i] <= 3);
        if(pass) pass = (ctx.d_quant[i] < 25);
    }

    SYCL_EXTERNAL void q11_agg(sycl::item<1> idx, Q11Context ctx, bool& pass) {
        if(pass) {
            size_t i = idx.get_id(0);
            auto ref = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*ctx.d_res);
            ref.fetch_add((uint64_t)ctx.d_price[i] * ctx.d_disc[i]);
        }
    }
}

template<typename T>
void load_column(const std::string& path, T* ptr, size_t n) {
    std::ifstream f(path, std::ios::binary); if (!f) exit(1);
    f.read(reinterpret_cast<char*>(ptr), n * sizeof(T));
}

int main(int argc, char** argv) {
    int repetitions = 10;
    std::string ssb_path = "/media/ssb/s100_columnar";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) repetitions = std::stoi(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) ssb_path = argv[++i];
    }
    size_t n = 600043265;
    sycl::queue q{sycl::default_selector_v};
    int *d_date = sycl::malloc_device<int>(n, q), *d_disc = sycl::malloc_device<int>(n, q), *d_quant = sycl::malloc_device<int>(n, q), *d_price = sycl::malloc_device<int>(n, q);
    uint64_t *d_res = sycl::malloc_device<uint64_t>(1, q);
    int *h_tmp = (int*)malloc(n*4);
    load_column(ssb_path + "/LINEORDER5", h_tmp, n); q.memcpy(d_date, h_tmp, n*4);
    load_column(ssb_path + "/LINEORDER11", h_tmp, n); q.memcpy(d_disc, h_tmp, n*4);
    load_column(ssb_path + "/LINEORDER8", h_tmp, n); q.memcpy(d_quant, h_tmp, n*4);
    load_column(ssb_path + "/LINEORDER9", h_tmp, n); q.memcpy(d_price, h_tmp, n*4).wait();
    free(h_tmp);

    Q11Context ctx{d_date, d_disc, d_quant, d_price, d_res};

    acpp_jit::dynamic_function_config cfg;
    cfg.define_as_call_sequence(&execute_q11_ops, {&q11_filter, &q11_agg});

    auto run_kernel = [&]() {
        q.parallel_for(sycl::range<1>{n}, cfg.apply([=](sycl::item<1> idx) {
            bool pass = true;
            execute_q11_ops(idx, ctx, pass);
        })).wait();
    };

    q.fill(d_res, 0ULL, 1).wait();
    run_kernel(); // Warmup and JIT trigger

    std::vector<double> times;
    for(int i=0; i<repetitions; ++i) {
        q.fill(d_res, 0ULL, 1).wait();
        auto start = std::chrono::high_resolution_clock::now();
        run_kernel();
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    double total = 0; for(auto t : times) total += t;
    double avg = total / times.size();
    double var = 0; for(auto t : times) var += (t-avg)*(t-avg);
    double stddev = std::sqrt(var/times.size());
    uint64_t final_res = 0; q.memcpy(&final_res, d_res, sizeof(uint64_t)).wait();
    std::cout << "Avg: " << avg << " ms, StdDev: " << stddev << " ms" << std::endl;
    std::cout << "Final result: " << final_res << std::endl;
    return 0;
}
