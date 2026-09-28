// HeteroGS CPU resource controls, applied to the pinned OpenSplat source only.
// Upstream and modifications are distributed under AGPL-3.0; see LICENSE.txt.
#pragma once
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace hgs {
inline int integer(const char *key, int fallback, int low, int high) {
    const char *raw = std::getenv(key);
    if (!raw) return fallback;
    const std::string s(raw);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string("Invalid resource setting: ") + key);
    const long long v = std::stoll(s);
    if (v < low || v > high)
        throw std::runtime_error(std::string("Out of range resource setting: ") + key);
    return static_cast<int>(v);
}
struct Limits {
    // Zero (only when unset) retains the upstream heuristic.
    int cpuThreads = integer("HGS_CPU_THREADS", 0, 1, 256);
    int rasterThreads = integer("HGS_RASTER_THREADS", 0, 1, 256);
    int gradientMiB = integer("HGS_GRADIENT_MIB", 64, 1, 4096);
    int imageWorkers = integer("HGS_IMAGE_WORKERS", 0, 1, 64);
    int cacheMiB = integer("HGS_IMAGE_CACHE_MIB", 0, 1, 4096);
    int imageSlots = integer("HGS_IMAGE_SLOTS", 0, 4, 128);
    int prepareImages = integer("HGS_PREPARE_IMAGES", 0, 1, 32);
    int strictGaussians = integer("HGS_STRICT_GAUSSIANS", 0, 0, 1);
};
inline const Limits &limits() { static const Limits v; return v; }
inline void describe() {
    const auto &v = limits();
    std::cout << "{\"resource_abi\":1,\"cpu_threads\":" << v.cpuThreads
              << ",\"raster_threads\":" << v.rasterThreads
              << ",\"gradient_mib\":" << v.gradientMiB
              << ",\"image_workers\":" << v.imageWorkers
              << ",\"image_cache_mib\":" << v.cacheMiB
              << ",\"image_slots\":" << v.imageSlots
              << ",\"prepare_images\":" << v.prepareImages
              << ",\"strict_gaussians\":" << v.strictGaussians << "}" << std::endl;
}
}
