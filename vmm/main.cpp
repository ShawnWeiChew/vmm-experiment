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
#include "../include/random_fill.cuh"
#include "../include/reference_matmul.cuh"
#include "../include/test_kernel.cuh"
#include "../mKernel/include/dist/distributed_buffer.cuh"
#include "../mKernel/include/operators/ag_gemm/ag_gemm_warp_specialized_globals.cuh"

constexpr int num_threads = 8;

RankState g_rs[8];
std::barrier<> bar(num_threads);
std::atomic<bool> test_failed{false};

// Host-side copies of each rank's random A shard / B, populated by
// fill_random in thread_func2 -- read after th.join() to run a reference
// matmul against whatever the GPU kernel writes into C.
std::vector<float> g_A_host[8];
std::vector<float> g_B_host[8];

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

    rank_ok = check_cuda(
        cudaMemcpy(local_mc_data, &initial_mc, sizeof(initial_mc), cudaMemcpyHostToDevice),
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

    rank_ok = check_cuda(
        launch_multicast_alias_fence(), "publish multicast writes to UC alias", device_num);

    MulticastTestData observed{};
    if (rank_ok) {
        rank_ok = check_cuda(
            cudaMemcpy(&observed, local_mc_data, sizeof(observed), cudaMemcpyDeviceToHost),
            "copy multicast-test result to host",
            device_num);
    }

    constexpr float expected_reduction = 3.0f;
    const float expected_red_target = static_cast<float>(device_num + 1) + kMulticastRedAddend;
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

void thread_func2(int device_num, std::barrier<>& bar) {
    bool rank_ok = check_cuda(cudaSetDevice(device_num), "cudaSetDevice", device_num);

    if (rank_ok && minfer_spmm_init(device_num, num_threads, device_num, 4096 * 4) != 0) {
        std::fprintf(stderr, "[rank %d] SP_MM initialization failed\n", device_num);
        rank_ok = false;
    }
    if (!rank_ok)
        test_failed.store(true);

    // Publish every rank's VMM mappings before looking up a peer address.
    bar.arrive_and_wait();

    // give every other rank its other peer's mappings
    for (int i = 0; i < num_threads; i++) {
        g_rs[device_num].peer_addresses[i] = g_rs[i].uc;
    }

    // have to try to recreate the local and distirbuted tensor from mKernel here
    using fg = ag_gemm_warp_specialized::fused_globals<128, 256, 2>;
    constexpr int N = 6400;
    constexpr int K = 7168;
    constexpr int M = 4096;
    constexpr int M_LOCAL = 4096 / num_threads;
    constexpr int logical_m = 6400;

    // allocate each device's tensor
    comm::bf16* B;
    check_cuda(cudaMalloc(&B, sizeof(comm::bf16) * N * K), "Malloc B", device_num);
    comm::bf16* C;
    check_cuda(cudaMalloc(&C, sizeof(comm::bf16) * M * N), "Malloc C", device_num);
    comm::bf16* A;
    check_cuda(cudaMalloc(&A, sizeof(comm::bf16) * M * K), "Malloc A", device_num);
    uint32_t* A_ready;
    check_cuda(cudaMalloc(&A_ready, sizeof(uint32_t) * 8), "Malloc A rdy", device_num);
    check_cuda(cudaMemset(A_ready, 0, sizeof(uint32_t) * 8), "Memset", device_num);

    // Distinct seeds per rank (offset by device_num) so no two ranks' B
    // shard or A shard end up with the same random data. fill_random reads
    // back through the same bf16 narrowing it wrote, so g_B_host/g_A_host
    // match the on-device bits exactly -- safe to use as-is for a host-side
    // reference matmul against whatever the kernel produces in C.
    //
    // B_view/A_shard_view are throwaway local_tensor wrappers used only to
    // drive fill_random -- they alias the same B/g_rs[device_num].uc device
    // memory that globals.B / globals.A (built below) will read.
    typename fg::B_local_tensor B_view =
        dist::local_tensor_from_data_ptr<typename fg::B_local_tensor>(
            reinterpret_cast<uint64_t>(B), 1, 1, N, K);
    typename fg::A_local_tensor A_shard_view =
        dist::local_tensor_from_data_ptr<typename fg::A_local_tensor>(
            g_rs[device_num].uc, 1, 1, M_LOCAL, K);
    g_B_host[device_num] = fill_random(B_view, /*seed=*/1000 * device_num + 2);
    g_A_host[device_num] = fill_random(A_shard_view, /*seed=*/1000 * device_num + 1);

    std::printf("[rank %d] filled A shard (%zu elems) and B (%zu elems) with random data\n",
                device_num,
                g_A_host[device_num].size(),
                g_B_host[device_num].size());

    // The kernel's AllGather needs every rank's shard filled before it reads
    // a peer's; the host reference below needs the same thing to safely read
    // g_A_host[1 - device_num].
    bar.arrive_and_wait();

    // Designated-initializer order must match fused_globals's member
    // declaration order (A, A_local_buf, B, C, A_copy_ready, A_copy_epoch,
    // dev_idx, M, N) -- C++20 rejects out-of-order designators.
    fg globals = {
        .A = dist::distributed_tensor_from_data_ptr<typename fg::A_distributed_tensor>(
            (uint64_t)g_rs[device_num].mc,
            (uint64_t*)g_rs[device_num].peer_addresses,
            1,
            1,
            M_LOCAL,
            K),
        .A_local_buf = dist::local_tensor_from_data_ptr<typename fg::A_local_tensor>(
            reinterpret_cast<uint64_t>(A), 1, 1, M, K),
        .B = dist::local_tensor_from_data_ptr<typename fg::B_local_tensor>(
            reinterpret_cast<uint64_t>(B), 1, 1, N, K),
        .C = dist::local_tensor_from_data_ptr<typename fg::C_local_tensor>(
            reinterpret_cast<uint64_t>(C), 1, 1, M, N),
        .A_copy_ready = A_ready,
        .A_copy_epoch = 1,
        .dev_idx = device_num,
        .M = M,
        .N = N,
    };

    launch_ag_gemm_warp_specialized<128, 256, 2, 5>(globals);
    if (!check_cuda(cudaDeviceSynchronize(), "sync after gemm launch", device_num)) {
        test_failed.store(true);
        return;
    }

    // Assemble the full gathered activation on the host: rank r contributed
    // rows [r*M_LOCAL, (r+1)*M_LOCAL). Both ranks crossed the barrier above,
    // so g_A_host[*] is fully populated by every thread at this point.
    std::vector<float> full_A_host;
    full_A_host.reserve(static_cast<size_t>(M) * K);
    for (int r = 0; r < num_threads; ++r) {
        full_A_host.insert(full_A_host.end(), g_A_host[r].begin(), g_A_host[r].end());
    }

    // This rank's own reference: full gathered A against this rank's own B
    // (each rank has independent random B in this harness), run on the GPU
    // via cuBLAS so the full M x N output is checked -- with 8 devices
    // instead of 2, this scales to 8 independent per-rank references
    // (num_threads controls it) rather than one shared answer.
    std::vector<float> reference =
        reference_matmul_cublas(full_A_host, M, K, g_B_host[device_num], N);

    std::vector<float> actual_C = read_back_as_float(C, static_cast<size_t>(M) * N);
    const bool matmul_ok = check_close(actual_C, reference, M, N);

    if (matmul_ok) {
        std::printf(
            "[rank %d] GEMM correctness check PASS (full %dx%d output)\n", device_num, M, N);
    } else {
        std::printf("[rank %d] GEMM correctness check FAILED\n", device_num);
        test_failed.store(true);
    }
}

int main() {
    // spawns 2 threads within the same process that are linked to different GPUs
    std::vector<std::thread> ths;

    for (int i = 0; i < num_threads; i++) {
        ths.emplace_back(thread_func2, i, std::ref(bar));
    }

    for (auto& th : ths) {
        th.join();
    }

    return test_failed.load() ? 1 : 0;
}
