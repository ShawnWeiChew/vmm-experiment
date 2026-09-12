#pragma once

#include <cuda_runtime.h>

#include <cstdint>

constexpr int kPeerTestElementsPerRank = 16;

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
    peer_copy_kernel<<<1, kPeerTestElementsPerRank>>>(
        local_buf,
        peer_buf,
        peer_rank);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;
    return cudaDeviceSynchronize();
}