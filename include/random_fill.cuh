#pragma once

/**
 * @file
 * @brief Fill a dist::local_tensor's device buffer with random floats, for
 * feeding a matmul reference check.
 *
 * Torch-free: works off the raw_ptr/numel()/dtype any dist::local_tensor
 * already exposes, so it's usable directly on fused_globals::A_tile/B_tile
 * shards or on a distributed_tensor's per-rank slice (`dist_tensor.gls[i]`).
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <type_traits>
#include <vector>

// Fills `t`'s device buffer with independent uniform randoms in [lo, hi),
// converted to T's on-device dtype (bf16/fp16/float all handled). `seed` is
// caller-controlled -- e.g. derive it from a rank index -- so each call
// (each B, each A_dist slice) gets distinct, reproducible data.
//
// Returns the values actually stored on device, read back through the same
// narrowing conversion (bf16 -> float, etc.), as a host reference for a
// matmul check -- so the reference matches the on-device data bit-for-bit
// rather than the pre-rounding randoms.
template <typename LocalTensor>
std::vector<float> fill_random(LocalTensor& t, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    using T = typename LocalTensor::dtype;
    const size_t n = t.numel();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);

    std::vector<T> device_side(n);
    std::vector<float> host_ref(n);
    for (size_t i = 0; i < n; ++i) {
        const float v = dist(rng);
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            device_side[i] = __float2bfloat16(v);
            host_ref[i] = __bfloat162float(device_side[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            device_side[i] = __float2half(v);
            host_ref[i] = __half2float(device_side[i]);
        } else {
            device_side[i] = static_cast<T>(v);
            host_ref[i] = static_cast<float>(device_side[i]);
        }
    }

    cudaError_t err = cudaMemcpy(t.raw_ptr, device_side.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "fill_random: cudaMemcpy failed -> %s\n", cudaGetErrorString(err));
    }
    return host_ref;
}

// Reads `count` elements of dtype T back from a raw device pointer (e.g. an
// output buffer a kernel just wrote), narrowing to float the same way
// fill_random does. Deliberately takes a raw pointer + count rather than a
// dist::local_tensor -- an output buffer's local_tensor view may have a
// shape that doesn't match its true element count (e.g. a C buffer sized
// M*N but whose local_tensor was built with the wrong column count), so
// giving the count explicitly avoids depending on that view being correct.
template <typename T>
std::vector<float> read_back_as_float(const T* device_ptr, size_t count) {
    std::vector<T> device_side(count);
    cudaError_t err =
        cudaMemcpy(device_side.data(), device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "read_back_as_float: cudaMemcpy failed -> %s\n", cudaGetErrorString(err));
    }

    std::vector<float> host(count);
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            host[i] = __bfloat162float(device_side[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            host[i] = __half2float(device_side[i]);
        } else {
            host[i] = static_cast<float>(device_side[i]);
        }
    }
    return host;
}
