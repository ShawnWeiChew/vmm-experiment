#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <thread>
#include <vector>

#include "../include/multimem.cuh"
#include "../include/test_kernel.cuh"

constexpr int num_threads = 2;

RankState g_rs[8];
std::barrier<> bar(num_threads);
std::atomic<bool> test_failed{false};

bool check_cuda(cudaError_t error, const char* operation, int rank) {
    if (error == cudaSuccess)
        return true;

    std::fprintf(stderr, "[rank %d] %s -> %s\n", rank, operation, cudaGetErrorString(error));
    return false;
}

void thread_func(int device_num, std::barrier<>& bar) {
    bool rank_ok = check_cuda(cudaSetDevice(device_num), "cudaSetDevice", device_num);

    if (rank_ok && minfer_spmm_init(device_num, num_threads, device_num, 4096 * 4) != 0) {
        std::fprintf(stderr, "[rank %d] SP_MM initialization failed\n", device_num);
        rank_ok = false;
    }
    if (!rank_ok)
        test_failed.store(true);

    // Publish every rank's VMM mappings before looking up a peer address.
    bar.arrive_and_wait();
    if (test_failed.load())
        return;

    constexpr std::size_t total_elements = num_threads * kPeerTestElementsPerRank;
    constexpr std::size_t total_bytes = total_elements * sizeof(std::uint32_t);

    std::vector<std::uint32_t> initial(total_elements, 0);
    for (int element = 0; element < kPeerTestElementsPerRank; ++element) {
        initial[device_num * kPeerTestElementsPerRank + element] =
            static_cast<std::uint32_t>(device_num + 1);
    }

    auto* local_buf = reinterpret_cast<std::uint32_t*>(g_rs[device_num].uc);
    rank_ok = check_cuda(cudaMemcpy(local_buf, initial.data(), total_bytes, cudaMemcpyHostToDevice),
                         "initialize peer-test buffer",
                         device_num);
    if (rank_ok) {
        rank_ok =
            check_cuda(cudaDeviceSynchronize(), "synchronize peer-test initialization", device_num);
    }
    if (!rank_ok)
        test_failed.store(true);

    // Remote reads cannot start until every peer has initialized its owned slot.
    bar.arrive_and_wait();
    if (test_failed.load())
        return;

    const int peer = 1 - device_num;
    const auto* peer_buf = reinterpret_cast<const std::uint32_t*>(g_rs[peer].uc);
    rank_ok = check_cuda(launch_test(local_buf, peer_buf, peer), "peer-copy kernel", device_num);

    std::vector<std::uint32_t> result(total_elements);
    if (rank_ok) {
        rank_ok =
            check_cuda(cudaMemcpy(result.data(), local_buf, total_bytes, cudaMemcpyDeviceToHost),
                       "copy peer-test result to host",
                       device_num);
    }

    for (int rank = 0; rank < num_threads && rank_ok; ++rank) {
        for (int element = 0; element < kPeerTestElementsPerRank; ++element) {
            const std::uint32_t actual = result[rank * kPeerTestElementsPerRank + element];
            const std::uint32_t expected = static_cast<std::uint32_t>(rank + 1);
            if (actual != expected) {
                std::fprintf(stderr,
                             "[rank %d] mismatch at rank %d element %d: "
                             "got %u, expected %u\n",
                             device_num,
                             rank,
                             element,
                             actual,
                             expected);
                rank_ok = false;
                break;
            }
        }
    }

    if (rank_ok) {
        std::printf("[rank %d] peer access PASS\n", device_num);
    } else {
        test_failed.store(true);
    }

    bar.arrive_and_wait();
    if (test_failed.load())
        return;

    auto* local_mc_data =
        reinterpret_cast<MulticastTestData*>(g_rs[device_num].uc + kMulticastTestOffset);
    MulticastTestData initial_mc{};
    initial_mc.reduce_input = static_cast<float>(device_num + 1);
    initial_mc.red_target = static_cast<float>(device_num + 1);

    rank_ok = check_cuda(cudaMemcpy(local_mc_data,
                                    &initial_mc,
                                    sizeof(initial_mc),
                                    cudaMemcpyHostToDevice),
                         "initialize multicast-test data",
                         device_num);
    if (rank_ok) {
        rank_ok = check_cuda(launch_multicast_alias_fence(),
                             "publish UC initialization to multicast alias",
                             device_num);
    }
    if (!rank_ok)
        test_failed.store(true);

    // Both physical allocations must be initialized and proxy-fenced before a
    // single GPU issues the multicast operations over the shared MC mapping.
    bar.arrive_and_wait();
    if (test_failed.load())
        return;

    if (device_num == 0) {
        auto* mc_data =
            reinterpret_cast<MulticastTestData*>(g_rs[device_num].mc + kMulticastTestOffset);
        rank_ok = check_cuda(launch_multicast_ops(mc_data, &local_mc_data->reduced_local),
                             "multicast operations kernel",
                             device_num);
        if (!rank_ok)
            test_failed.store(true);
    }

    bar.arrive_and_wait();
    if (test_failed.load())
        return;

    rank_ok = check_cuda(launch_multicast_alias_fence(),
                         "publish multicast writes to UC alias",
                         device_num);

    MulticastTestData observed{};
    if (rank_ok) {
        rank_ok = check_cuda(cudaMemcpy(&observed,
                                        local_mc_data,
                                        sizeof(observed),
                                        cudaMemcpyDeviceToHost),
                             "copy multicast-test result to host",
                             device_num);
    }

    constexpr float expected_reduction = 3.0f;
    const float expected_red_target =
        static_cast<float>(device_num + 1) + kMulticastRedAddend;
    if (rank_ok && observed.store_output != expected_reduction) {
        std::fprintf(stderr,
                     "[rank %d] multimem.st mismatch: got %.1f, expected %.1f\n",
                     device_num,
                     observed.store_output,
                     expected_reduction);
        rank_ok = false;
    }
    if (rank_ok && observed.red_target != expected_red_target) {
        std::fprintf(stderr,
                     "[rank %d] multimem.red mismatch: got %.1f, expected %.1f\n",
                     device_num,
                     observed.red_target,
                     expected_red_target);
        rank_ok = false;
    }
    if (rank_ok && device_num == 0 && observed.reduced_local != expected_reduction) {
        std::fprintf(stderr,
                     "[rank 0] multimem.ld_reduce mismatch: got %.1f, expected %.1f\n",
                     observed.reduced_local,
                     expected_reduction);
        rank_ok = false;
    }

    if (rank_ok) {
        if (device_num == 0)
            std::printf("[rank 0] multimem.ld_reduce PASS (1.0 + 2.0 = 3.0)\n");
        std::printf("[rank %d] multimem.st PASS (3.0), multimem.red PASS (%.1f)\n",
                    device_num,
                    expected_red_target);
    } else {
        test_failed.store(true);
    }
}

int main() {
    // spawns 2 threads within the same process that are linked to different GPUs
    std::vector<std::thread> ths;

    for (int i = 0; i < num_threads; i++) {
        ths.emplace_back(thread_func, i, std::ref(bar));
    }

    for (auto& th : ths) {
        th.join();
    }

    return test_failed.load() ? 1 : 0;
}
