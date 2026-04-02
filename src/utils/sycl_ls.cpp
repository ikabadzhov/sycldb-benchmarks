#include "sycl_device.hpp"

#include <iostream>

int main() {
    const auto devices = sycldb::enumerate_devices();
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& device = devices[i];
        const auto platform = device.get_platform();
        std::cout
            << "[" << i << "] "
            << device.get_info<sycl::info::device::name>()
            << " | platform=" << platform.get_info<sycl::info::platform::name>()
            << " | vendor=" << device.get_info<sycl::info::device::vendor>()
            << std::endl;
    }
    return 0;
}
