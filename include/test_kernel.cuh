#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

constexpr int kPeerTestElementsPerRank = 16;
constexpr std::size_t kMulticastTestOffset = 4096;
constexpr float kMulticastRedAddend = 10.0f;

struct alignas(16) MulticastTestData {
    float reduce_input;
    float store_output;
    float red_target;
    float reduced_local;
};

__global__ void peer_copy_kernel(std::uint32_t* local_buf,
                                 const std::uint32_t* peer_buf,
                                 int peer_rank) {
    const int element = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (element >= kPeerTestElementsPerRank)
        return;

    const int index = peer_rank * kPeerTestElementsPerRank + element;
    local_buf[index] = peer_buf[index];
}

inline cudaError_t launch_test(std::uint32_t* local_buf,
                               const std::uint32_t* peer_buf,
                               int peer_rank) {
    peer_copy_kernel<<<1, kPeerTestElementsPerRank>>>(local_buf, peer_buf, peer_rank);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;
    return cudaDeviceSynchronize();
}

__global__ void multicast_alias_fence_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("fence.proxy.alias;" ::: "memory");
}

__global__ void multicast_ops_kernel(MulticastTestData* mc_data, float* reduced_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    float reduced;
    const float* reduce_input = &mc_data->reduce_input;
    float* store_output = &mc_data->store_output;
    float* red_target = &mc_data->red_target;

    asm volatile("multimem.ld_reduce.acquire.sys.global.add.f32 %0, [%1];"
                 : "=f"(reduced)
                 : "l"(reduce_input)
                 : "memory");

    const std::uint32_t reduced_bits = __float_as_uint(reduced);
    asm volatile("multimem.st.release.sys.global.b32 [%0], %1;"
                 :
                 : "l"(store_output), "r"(reduced_bits)
                 : "memory");
    asm volatile("multimem.red.release.sys.global.add.f32 [%0], %1;"
                 :
                 : "l"(red_target), "f"(kMulticastRedAddend)
                 : "memory");

    // The result is stored through the ordinary alias, so order the multicast
    // proxy operations before switching address-space aliases.
    asm volatile("fence.proxy.alias;" ::: "memory");
    *reduced_out = reduced;
}

inline cudaError_t launch_multicast_alias_fence() {
    multicast_alias_fence_kernel<<<1, 1>>>();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;
    return cudaDeviceSynchronize();
}

inline cudaError_t launch_multicast_ops(MulticastTestData* mc_data, float* reduced_out) {
    multicast_ops_kernel<<<1, 1>>>(mc_data, reduced_out);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;
    return cudaDeviceSynchronize();
}
