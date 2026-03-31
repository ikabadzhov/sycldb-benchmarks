#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>

/**
 * SSB Q1.1 Standalone Benchmark (Hardcoded SYCL Coalesced / Vector Fetch)
 * Uses sycl::vec<int, 4> natively for perfect 128-bit coalesced memory 
 * transactions across the GPU warp.
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

class Q11CoalescedKernel {
private:
    const sycl::int4 *lo_orderdate;
    const sycl::int4 *lo_discount;
    const sycl::int4 *lo_quantity;
    const sycl::int4 *lo_extendedprice;
    uint64_t *agg_result;
    size_t num_vec_items;

public:
    Q11CoalescedKernel(
        const sycl::int4 *date, const sycl::int4 *disc, const sycl::int4 *quant, const sycl::int4 *price, uint64_t *res, size_t n_vecs)
        : lo_orderdate(date), lo_discount(disc), lo_quantity(quant), lo_extendedprice(price), agg_result(res), num_vec_items(n_vecs)
    {}

    void operator()(sycl::id<1> idx) const {
        int i = idx[0];
        if (i >= num_vec_items) return;

        bool f[4] = {true, true, true, true};
        sycl::int4 date = lo_orderdate[i];
        
        f[0] = logical(AND, f[0], compare(GE, date.x(), 19930101)); f[0] = logical(AND, f[0], compare(LE, date.x(), 19931231));
        f[1] = logical(AND, f[1], compare(GE, date.y(), 19930101)); f[1] = logical(AND, f[1], compare(LE, date.y(), 19931231));
        f[2] = logical(AND, f[2], compare(GE, date.z(), 19930101)); f[2] = logical(AND, f[2], compare(LE, date.z(), 19931231));
        f[3] = logical(AND, f[3], compare(GE, date.w(), 19930101)); f[3] = logical(AND, f[3], compare(LE, date.w(), 19931231));

        if (f[0] || f[1] || f[2] || f[3]) {
            sycl::int4 disc = lo_discount[i];
            if (f[0]) f[0] = logical(AND, compare(GE, disc.x(), 1), compare(LE, disc.x(), 3));
            if (f[1]) f[1] = logical(AND, compare(GE, disc.y(), 1), compare(LE, disc.y(), 3));
            if (f[2]) f[2] = logical(AND, compare(GE, disc.z(), 1), compare(LE, disc.z(), 3));
            if (f[3]) f[3] = logical(AND, compare(GE, disc.w(), 1), compare(LE, disc.w(), 3));

            if (f[0] || f[1] || f[2] || f[3]) {
                sycl::int4 quant = lo_quantity[i];
                if (f[0]) f[0] = compare(LT, quant.x(), 25);
                if (f[1]) f[1] = compare(LT, quant.y(), 25);
                if (f[2]) f[2] = compare(LT, quant.z(), 25);
                if (f[3]) f[3] = compare(LT, quant.w(), 25);

                uint64_t local_sum = 0;
                if (f[0] || f[1] || f[2] || f[3]) {
                    sycl::int4 price = lo_extendedprice[i];
                    if (f[0]) local_sum += element_operation((uint64_t)price.x(), (uint64_t)disc.x(), BinaryOp::Multiply);
                    if (f[1]) local_sum += element_operation((uint64_t)price.y(), (uint64_t)disc.y(), BinaryOp::Multiply);
                    if (f[2]) local_sum += element_operation((uint64_t)price.z(), (uint64_t)disc.z(), BinaryOp::Multiply);
                    if (f[3]) local_sum += element_operation((uint64_t)price.w(), (uint64_t)disc.w(), BinaryOp::Multiply);
                }

                if (local_sum > 0) {
                    sycl::atomic_ref<
                        uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space
                    > sum_obj(*agg_result);
                    sum_obj.fetch_add(local_sum);
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
    int repetitions = 3;
    std::string ssb_path = "/media/ssb/sf100_columnar";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) repetitions = std::stoi(argv[++i]);
        else if (arg == "-p" && i + 1 < argc) ssb_path = argv[++i];
    }

    sycl::queue q{sycl::default_selector_v};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    size_t n = get_file_rows(ssb_path + "/LINEORDER5");
    if (n == 0 && ssb_path == "/media/ssb/sf100_columnar") {
        ssb_path = "/media/ssb/s100_columnar";
        n = get_file_rows(ssb_path + "/LINEORDER5");
    }

    if (n == 0) { std::cerr << "Could not find ssb files in " << ssb_path << std::endl; return 1; }
    
    // Ensure padding to div by 4
    size_t padded_n = ((n + 3) / 4) * 4;
    std::cout << "Table [LINEORDER] size: " << n << " rows (padded to " << padded_n << ")" << std::endl;

    int* h_date = (int*)malloc(padded_n * sizeof(int));
    int* h_disc = (int*)malloc(padded_n * sizeof(int));
    int* h_quant = (int*)malloc(padded_n * sizeof(int));
    int* h_price = (int*)malloc(padded_n * sizeof(int));

    load_column(ssb_path + "/LINEORDER5", h_date, n);
    load_column(ssb_path + "/LINEORDER11", h_disc, n);
    load_column(ssb_path + "/LINEORDER8", h_quant, n);
    load_column(ssb_path + "/LINEORDER9", h_price, n);

    for(size_t i=n; i<padded_n; ++i) { h_date[i]=0; h_disc[i]=0; h_quant[i]=0; h_price[i]=0; }

    sycl::int4* d_date = sycl::malloc_device<sycl::int4>(padded_n / 4, q);
    sycl::int4* d_disc = sycl::malloc_device<sycl::int4>(padded_n / 4, q);
    sycl::int4* d_quant = sycl::malloc_device<sycl::int4>(padded_n / 4, q);
    sycl::int4* d_price = sycl::malloc_device<sycl::int4>(padded_n / 4, q);
    uint64_t* d_res = sycl::malloc_device<uint64_t>(1, q);

    q.memcpy(d_date, h_date, padded_n * sizeof(int));
    q.memcpy(d_disc, h_disc, padded_n * sizeof(int));
    q.memcpy(d_quant, h_quant, padded_n * sizeof(int));
    q.memcpy(d_price, h_price, padded_n * sizeof(int)).wait();

    free(h_date); free(h_disc); free(h_quant); free(h_price);

    size_t n_vecs = padded_n / 4;

    auto run_kernel = [&]() {
        Q11CoalescedKernel kernel(d_date, d_disc, d_quant, d_price, d_res, n_vecs);
        q.parallel_for(sycl::range<1>(n_vecs), kernel).wait();
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

    double avg = 0; for(auto t : times) avg += t; avg /= times.size();
    double var = 0; for(auto t : times) var += (t-avg)*(t-avg); double stddev = std::sqrt(var/times.size());

    uint64_t final_res = 0; q.memcpy(&final_res, d_res, 8).wait();
    std::cout << "Execution time over " << repetitions << " repetitions - Avg: " << avg << " ms, StdDev: " << stddev << " ms" << std::endl;
    std::cout << "Final result: " << final_res << std::endl;

    sycl::free(d_date, q); sycl::free(d_disc, q); sycl::free(d_quant, q); sycl::free(d_price, q); sycl::free(d_res, q);
    return 0;
}
