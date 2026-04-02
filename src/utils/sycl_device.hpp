#pragma once

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace sycldb {

inline std::vector<sycl::device> enumerate_devices() {
    std::vector<sycl::device> devices;
    for (const auto& platform : sycl::platform::get_platforms()) {
        auto platform_devices = platform.get_devices();
        devices.insert(devices.end(), platform_devices.begin(), platform_devices.end());
    }
    return devices;
}

inline int find_requested_device_id(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-d") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after -d" << std::endl;
                std::exit(1);
            }
            return std::stoi(argv[i + 1]);
        }
    }
    return -1;
}

inline sycl::queue make_queue_from_args(int argc, char** argv) {
    const int device_id = find_requested_device_id(argc, argv);
    if (device_id < 0) {
        return sycl::queue{sycl::default_selector_v};
    }

    const auto devices = enumerate_devices();
    if (device_id >= static_cast<int>(devices.size())) {
        std::cerr
            << "Requested device id " << device_id << " is out of range (found "
            << devices.size() << " devices)" << std::endl;
        std::exit(1);
    }
    return sycl::queue{devices[device_id]};
}

}  // namespace sycldb
