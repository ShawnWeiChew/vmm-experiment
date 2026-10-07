#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

#include "../include/multimem.cuh"

namespace {

constexpr int kWorldSize = 8;
constexpr int kK = 7168;
constexpr int kWarmupIterations = 3;
constexpr int kIterations = 20;
constexpr std::array<int, 7> kMValues = {2048, 3072, 3548, 4096, 8192, 16384, 32768};

constexpr std::size_t kElementBytes = sizeof(__nv_bfloat16);
constexpr std::size_t tensor_bytes(int m) {
    return static_cast<std::size_t>(m) * kK * kElementBytes;
}
constexpr std::size_t shard_bytes(int m) {
    return tensor_bytes(m) / kWorldSize;
}
constexpr std::size_t kMaxTensorBytes = tensor_bytes(kMValues.back());
constexpr std::size_t kMaxShardBytes = shard_bytes(kMValues.back());

static_assert(kMaxTensorBytes <= kSlab);
static_assert(kMaxTensorBytes % sizeof(std::uint64_t) == 0);
static_assert(kMaxShardBytes % sizeof(std::uint64_t) == 0);

struct Timings {
    float peer_pull_ms = 0.0f;
    float single_mc_push_ms = 0.0f;
    float concurrent_mc_push_ms = 0.0f;
    float single_peer_push_ms = 0.0f;
    float quack_ag_push_ms = 0.0f;
};

std::barrier<> g_barrier(kWorldSize);
std::atomic<bool> g_failed{false};
std::array<std::array<Timings, kWorldSize>, kMValues.size()> g_timings{};
std::array<void*, kWorldSize> g_sources{};

bool check_cuda(cudaError_t error, const char* operation, int rank) {
    if (error == cudaSuccess)
        return true;

    std::fprintf(stderr,
                 "[rank %d] %s failed: %s\n",
                 rank,
                 operation,
                 cudaGetErrorString(error));
    g_failed.store(true, std::memory_order_relaxed);
    return false;
}

__global__ void multicast_alias_fence_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("fence.proxy.alias;" ::: "memory");
}

__global__ void verify_gather_kernel(const std::uint64_t* gathered,
                                     std::size_t words_per_shard,
                                     unsigned long long* mismatches) {
    constexpr std::uint64_t kRepeatedByte = 0x0101010101010101ull;
    const std::size_t total_words = words_per_shard * kWorldSize;

    for (std::size_t word = blockIdx.x * blockDim.x + threadIdx.x;
         word < total_words;
         word += gridDim.x * blockDim.x) {
        const int source_rank = static_cast<int>(word / words_per_shard);
        const std::uint64_t expected =
            static_cast<std::uint64_t>(source_rank + 1) * kRepeatedByte;
        if (gathered[word] != expected)
            atomicAdd(mismatches, 1ull);
    }
}

__global__ void verify_pattern_kernel(const std::uint64_t* data,
                                      std::size_t word_count,
                                      std::uint64_t expected,
                                      unsigned long long* mismatches) {
    for (std::size_t word = blockIdx.x * blockDim.x + threadIdx.x;
         word < word_count;
         word += gridDim.x * blockDim.x) {
        if (data[word] != expected)
            atomicAdd(mismatches, 1ull);
    }
}

void* device_pointer(CUdeviceptr pointer) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
}

bool enable_peer_access(int rank) {
    for (int peer = 0; peer < kWorldSize; ++peer) {
        if (peer == rank)
            continue;

        int can_access = 0;
        if (!check_cuda(cudaDeviceCanAccessPeer(&can_access, rank, peer),
                        "query peer access",
                        rank)) {
            return false;
        }
        if (!can_access) {
            std::fprintf(stderr, "[rank %d] cannot access peer GPU %d\n", rank, peer);
            g_failed.store(true, std::memory_order_relaxed);
            return false;
        }

        const cudaError_t error = cudaDeviceEnablePeerAccess(peer, 0);
        if (error == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
        } else if (!check_cuda(error, "enable peer access", rank)) {
            return false;
        }
    }
    return true;
}

bool enqueue_peer_pull(std::size_t bytes_per_shard, cudaStream_t stream) {
    // Rank 0 pulls exactly one shard from rank 1 into rank 0's local VMM
    // allocation. No other GPU issues a copy during this experiment.
    const CUdeviceptr rank_1_slot = bytes_per_shard;
    return check_cuda(cudaMemcpyAsync(device_pointer(g_rs[0].uc + rank_1_slot),
                                      device_pointer(g_rs[1].uc + rank_1_slot),
                                      bytes_per_shard,
                                      cudaMemcpyDeviceToDevice,
                                      stream),
                      "enqueue rank 0 <- rank 1 peer pull",
                      0);
}

bool enqueue_single_multicast_push(std::size_t bytes_per_shard, cudaStream_t stream) {
    // Rank 0 copies the same amount of local data to the multicast mapping.
    // The one copy is broadcast into the physical allocation of all 8 ranks.
    return check_cuda(cudaMemcpyAsync(device_pointer(g_rs[0].mc),
                                      g_sources[0],
                                      bytes_per_shard,
                                      cudaMemcpyDeviceToDevice,
                                      stream),
                      "enqueue rank 0 multicast push",
                      0);
}

bool enqueue_multicast_push(int rank, std::size_t bytes_per_shard, cudaStream_t stream) {
    const CUdeviceptr destination =
        g_rs[rank].mc + static_cast<std::size_t>(rank) * bytes_per_shard;
    return check_cuda(cudaMemcpyAsync(device_pointer(destination),
                                      g_sources[rank],
                                      bytes_per_shard,
                                      cudaMemcpyDeviceToDevice,
                                      stream),
                      "enqueue multicast push",
                      rank);
}

bool enqueue_single_peer_push(std::size_t bytes_per_shard, cudaStream_t stream) {
    // Run on rank 1: push rank 1's staged shard into rank 0's rank-1 slot.
    const CUdeviceptr rank_1_slot = bytes_per_shard;
    return check_cuda(cudaMemcpyAsync(device_pointer(g_rs[0].uc + rank_1_slot),
                                      device_pointer(g_rs[1].uc + rank_1_slot),
                                      bytes_per_shard,
                                      cudaMemcpyDeviceToDevice,
                                      stream),
                      "enqueue rank 1 -> rank 0 peer push",
                      1);
}

bool enqueue_quack_ag_push(int rank,
                           std::size_t bytes_per_shard,
                           cudaStream_t stream) {
    // QuACK CE-push order: the owner sends its staged shard to rank-1 first,
    // then rank-2, and so on. The local slot is staged before timing.
    const CUdeviceptr source =
        g_rs[rank].uc + static_cast<std::size_t>(rank) * bytes_per_shard;
    for (int step = 1; step < kWorldSize; ++step) {
        const int destination_rank = (rank - step + kWorldSize) % kWorldSize;
        const CUdeviceptr destination =
            g_rs[destination_rank].uc + static_cast<std::size_t>(rank) * bytes_per_shard;
        if (!check_cuda(cudaMemcpyAsync(device_pointer(destination),
                                        device_pointer(source),
                                        bytes_per_shard,
                                        cudaMemcpyDeviceToDevice,
                                        stream),
                        "enqueue QuACK peer push",
                        rank)) {
            return false;
        }
    }
    return true;
}

bool prepare_quack_buffer(int rank,
                          std::size_t bytes_per_tensor,
                          std::size_t bytes_per_shard,
                          cudaStream_t stream) {
    const CUdeviceptr local_slot =
        g_rs[rank].uc + static_cast<std::size_t>(rank) * bytes_per_shard;
    return check_cuda(cudaMemsetAsync(device_pointer(g_rs[rank].uc),
                                      0,
                                      bytes_per_tensor,
                                      stream),
                      "clear QuACK gather buffer",
                      rank) &&
           check_cuda(cudaMemcpyAsync(device_pointer(local_slot),
                                      g_sources[rank],
                                      bytes_per_shard,
                                      cudaMemcpyDeviceToDevice,
                                      stream),
                      "stage local QuACK shard",
                      rank) &&
           check_cuda(cudaStreamSynchronize(stream),
                      "synchronize QuACK buffer preparation",
                      rank);
}

bool verify_one_shard(int rank,
                      cudaStream_t stream,
                      unsigned long long* device_mismatches,
                      std::size_t bytes_per_shard,
                      int destination_slot,
                      int source_rank,
                      bool multicast_write) {
    if (multicast_write) {
        multicast_alias_fence_kernel<<<1, 1, 0, stream>>>();
        if (!check_cuda(cudaGetLastError(), "launch multicast alias fence", rank))
            return false;
    }

    if (!check_cuda(cudaMemsetAsync(device_mismatches, 0, sizeof(*device_mismatches), stream),
                    "clear mismatch count",
                    rank)) {
        return false;
    }

    constexpr std::uint64_t kRepeatedByte = 0x0101010101010101ull;
    constexpr int kThreads = 256;
    constexpr int kBlocks = 1024;
    verify_pattern_kernel<<<kBlocks, kThreads, 0, stream>>>(
        reinterpret_cast<const std::uint64_t*>(
            static_cast<std::uintptr_t>(
                g_rs[rank].uc + static_cast<std::size_t>(destination_slot) * bytes_per_shard)),
        bytes_per_shard / sizeof(std::uint64_t),
        static_cast<std::uint64_t>(source_rank + 1) * kRepeatedByte,
        device_mismatches);
    if (!check_cuda(cudaGetLastError(), "launch one-shard verification", rank))
        return false;

    unsigned long long mismatches = 0;
    if (!check_cuda(cudaMemcpyAsync(&mismatches,
                                    device_mismatches,
                                    sizeof(mismatches),
                                    cudaMemcpyDeviceToHost,
                                    stream),
                    "copy mismatch count",
                    rank) ||
        !check_cuda(cudaStreamSynchronize(stream), "synchronize verification", rank)) {
        return false;
    }

    if (mismatches != 0) {
        std::fprintf(stderr,
                     "[rank %d] one-shard verification failed: "
                     "%llu mismatched 64-bit words\n",
                     rank,
                     mismatches);
        g_failed.store(true, std::memory_order_relaxed);
        return false;
    }
    return true;
}

bool verify_gather(int rank,
                   cudaStream_t stream,
                   unsigned long long* device_mismatches,
                   std::size_t bytes_per_shard,
                   bool multicast_writes) {
    if (multicast_writes) {
        multicast_alias_fence_kernel<<<1, 1, 0, stream>>>();
        if (!check_cuda(cudaGetLastError(), "launch multicast alias fence", rank))
            return false;
    }

    if (!check_cuda(cudaMemsetAsync(device_mismatches, 0, sizeof(*device_mismatches), stream),
                    "clear mismatch count",
                    rank)) {
        return false;
    }

    constexpr int kThreads = 256;
    constexpr int kBlocks = 1024;
    verify_gather_kernel<<<kBlocks, kThreads, 0, stream>>>(
        reinterpret_cast<const std::uint64_t*>(
            static_cast<std::uintptr_t>(g_rs[rank].uc)),
        bytes_per_shard / sizeof(std::uint64_t),
        device_mismatches);
    if (!check_cuda(cudaGetLastError(), "launch gather verification", rank))
        return false;

    unsigned long long mismatches = 0;
    if (!check_cuda(cudaMemcpyAsync(&mismatches,
                                    device_mismatches,
                                    sizeof(mismatches),
                                    cudaMemcpyDeviceToHost,
                                    stream),
                    "copy mismatch count",
                    rank) ||
        !check_cuda(cudaStreamSynchronize(stream), "synchronize verification", rank)) {
        return false;
    }

    if (mismatches != 0) {
        std::fprintf(stderr,
                     "[rank %d] verification failed: %llu mismatched 64-bit words\n",
                     rank,
                     mismatches);
        g_failed.store(true, std::memory_order_relaxed);
        return false;
    }
    return true;
}

template <typename Enqueue>
bool warm_up(int rank, cudaStream_t stream, Enqueue enqueue) {
    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        if (!enqueue())
            return false;
    }
    return check_cuda(cudaStreamSynchronize(stream), "synchronize warmup", rank);
}

template <typename Enqueue>
bool time_repeated(int rank,
                   cudaStream_t stream,
                   cudaEvent_t start,
                   cudaEvent_t stop,
                   Enqueue enqueue,
                   float& elapsed_ms) {
    if (!check_cuda(cudaEventRecord(start, stream), "record start event", rank))
        return false;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        if (!enqueue())
            return false;
    }
    if (!check_cuda(cudaEventRecord(stop, stream), "record stop event", rank) ||
        !check_cuda(cudaEventSynchronize(stop), "synchronize stop event", rank) ||
        !check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "read elapsed time", rank)) {
        return false;
    }

    elapsed_ms /= kIterations;
    return true;
}

void benchmark_rank(int rank) {
    bool rank_ok = check_cuda(cudaSetDevice(rank), "cudaSetDevice", rank);
    if (rank_ok && minfer_spmm_init(rank, kWorldSize, rank, kMaxTensorBytes) != 0) {
        std::fprintf(stderr, "[rank %d] multicast VMM initialization failed\n", rank);
        g_failed.store(true, std::memory_order_relaxed);
        rank_ok = false;
    }
    if (rank_ok)
        rank_ok = enable_peer_access(rank);

    // rank_setup publishes the VMM mappings. Do not let any rank use a peer
    // address until all eight mappings exist.
    g_barrier.arrive_and_wait();
    if (g_failed.load(std::memory_order_relaxed))
        return;

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    unsigned long long* device_mismatches = nullptr;

    rank_ok = check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                         "create stream",
                         rank) &&
              check_cuda(cudaEventCreate(&start), "create start event", rank) &&
              check_cuda(cudaEventCreate(&stop), "create stop event", rank) &&
              check_cuda(cudaMalloc(&device_mismatches, sizeof(*device_mismatches)),
                         "allocate mismatch count",
                         rank) &&
              check_cuda(cudaMalloc(&g_sources[rank], kMaxShardBytes),
                         "allocate source shard",
                         rank);

    g_barrier.arrive_and_wait();
    if (g_failed.load(std::memory_order_relaxed)) {
        if (g_sources[rank])
            cudaFree(g_sources[rank]);
        if (device_mismatches)
            cudaFree(device_mismatches);
        if (stop)
            cudaEventDestroy(stop);
        if (start)
            cudaEventDestroy(start);
        if (stream)
            cudaStreamDestroy(stream);
        return;
    }

    for (std::size_t shape_index = 0; shape_index < kMValues.size(); ++shape_index) {
        const int m = kMValues[shape_index];
        const std::size_t bytes_per_tensor = tensor_bytes(m);
        const std::size_t bytes_per_shard = shard_bytes(m);
        Timings& timings = g_timings[shape_index][rank];

        // Each rank owns one distinct source shard. The three timed cases all
        // copy exactly bytes_per_shard bytes per active GPU.
        rank_ok = check_cuda(cudaMemsetAsync(device_pointer(g_rs[rank].uc),
                                             0,
                                             bytes_per_tensor,
                                             stream),
                             "clear destination",
                             rank) &&
                  check_cuda(cudaMemsetAsync(g_sources[rank],
                                             rank + 1,
                                             bytes_per_shard,
                                             stream),
                             "initialize source shard",
                             rank) &&
                  check_cuda(cudaMemcpyAsync(
                                 device_pointer(
                                     g_rs[rank].uc +
                                     static_cast<std::size_t>(rank) * bytes_per_shard),
                                 g_sources[rank],
                                 bytes_per_shard,
                                 cudaMemcpyDeviceToDevice,
                                 stream),
                             "stage local shard for peer pull",
                             rank) &&
                  check_cuda(cudaStreamSynchronize(stream),
                             "synchronize initialization",
                             rank);
        g_barrier.arrive_and_wait();
        if (!rank_ok || g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        // Case 1: rank 0 pulls one shard from rank 1. Every other rank is idle.
        if (rank == 0) {
            rank_ok = warm_up(
                          rank,
                          stream,
                          [&] { return enqueue_peer_pull(bytes_per_shard, stream); }) &&
                      time_repeated(rank,
                                    stream,
                                    start,
                                    stop,
                                    [&] { return enqueue_peer_pull(bytes_per_shard, stream); },
                                    timings.peer_pull_ms);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (rank == 0 && !g_failed.load(std::memory_order_relaxed)) {
            rank_ok = verify_one_shard(
                rank, stream, device_mismatches, bytes_per_shard, 1, 1, false);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        // Case 2: rank 0 pushes one local shard to multicast memory. This is
        // the direct single-device comparison with the peer pull above.
        rank_ok = check_cuda(cudaMemsetAsync(device_pointer(g_rs[rank].uc),
                                             0,
                                             bytes_per_tensor,
                                             stream),
                             "clear destination before single multicast",
                             rank) &&
                  check_cuda(cudaStreamSynchronize(stream),
                             "synchronize destination clear",
                             rank);
        g_barrier.arrive_and_wait();
        if (!rank_ok || g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        if (rank == 0) {
            rank_ok =
                warm_up(rank,
                        stream,
                        [&] { return enqueue_single_multicast_push(bytes_per_shard, stream); }) &&
                time_repeated(
                    rank,
                    stream,
                    start,
                    stop,
                    [&] { return enqueue_single_multicast_push(bytes_per_shard, stream); },
                    timings.single_mc_push_ms);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (!g_failed.load(std::memory_order_relaxed)) {
            rank_ok =
                verify_one_shard(rank, stream, device_mismatches, bytes_per_shard, 0, 0, true);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        // Case 3: every GPU performs the same multicast push as case 2 at the
        // same time. Disjoint destinations avoid data races between senders.
        rank_ok = check_cuda(cudaMemsetAsync(device_pointer(g_rs[rank].uc),
                                             0,
                                             bytes_per_tensor,
                                             stream),
                             "clear destination before concurrent multicast",
                             rank) &&
                  check_cuda(cudaStreamSynchronize(stream),
                             "synchronize destination clear",
                             rank);
        g_barrier.arrive_and_wait();
        if (!rank_ok || g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        rank_ok = warm_up(
            rank,
            stream,
            [&] { return enqueue_multicast_push(rank, bytes_per_shard, stream); });
        if (!rank_ok)
            g_failed.store(true, std::memory_order_relaxed);
        g_barrier.arrive_and_wait();
        if (!g_failed.load(std::memory_order_relaxed)) {
            rank_ok = time_repeated(
                rank,
                stream,
                start,
                stop,
                [&] { return enqueue_multicast_push(rank, bytes_per_shard, stream); },
                timings.concurrent_mc_push_ms);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (!g_failed.load(std::memory_order_relaxed)) {
            rank_ok = verify_gather(rank, stream, device_mismatches, bytes_per_shard, true);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        // Case 4: rank 1 owns the copy and pushes its staged shard into rank
        // 0's local buffer. Cases 1 and 4 therefore use identical endpoints
        // and bytes, changing only which GPU's copy engine owns the transfer.
        rank_ok = prepare_quack_buffer(
            rank, bytes_per_tensor, bytes_per_shard, stream);
        g_barrier.arrive_and_wait();
        if (!rank_ok || g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        if (rank == 1) {
            rank_ok =
                warm_up(rank,
                        stream,
                        [&] { return enqueue_single_peer_push(bytes_per_shard, stream); }) &&
                time_repeated(rank,
                              stream,
                              start,
                              stop,
                              [&] { return enqueue_single_peer_push(bytes_per_shard, stream); },
                              timings.single_peer_push_ms);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (rank == 0 && !g_failed.load(std::memory_order_relaxed)) {
            rank_ok = verify_one_shard(
                rank, stream, device_mismatches, bytes_per_shard, 1, 1, false);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        // Case 5: QuACK CE-push all-gather. Every rank's own shard is already
        // staged locally; all ranks concurrently send it to their 7 peers in
        // reverse-ring order. Local staging and arrival-flag writes are
        // intentionally outside timing so this remains a data-memcpy test.
        rank_ok = prepare_quack_buffer(
            rank, bytes_per_tensor, bytes_per_shard, stream);
        g_barrier.arrive_and_wait();
        if (!rank_ok || g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        rank_ok = warm_up(
            rank,
            stream,
            [&] { return enqueue_quack_ag_push(rank, bytes_per_shard, stream); });
        if (!rank_ok)
            g_failed.store(true, std::memory_order_relaxed);
        g_barrier.arrive_and_wait();
        if (!g_failed.load(std::memory_order_relaxed)) {
            rank_ok = time_repeated(
                rank,
                stream,
                start,
                stop,
                [&] { return enqueue_quack_ag_push(rank, bytes_per_shard, stream); },
                timings.quack_ag_push_ms);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (!g_failed.load(std::memory_order_relaxed)) {
            rank_ok = verify_gather(rank, stream, device_mismatches, bytes_per_shard, false);
            if (!rank_ok)
                g_failed.store(true, std::memory_order_relaxed);
        }
        g_barrier.arrive_and_wait();
        if (g_failed.load(std::memory_order_relaxed))
            goto cleanup;

        if (rank == 0)
            std::printf("completed M=%d, K=%d\n", m, kK);
    }

cleanup:
    cudaFree(g_sources[rank]);
    cudaFree(device_mismatches);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
}

struct Summary {
    float minimum;
    float average;
    float maximum;
};

template <typename Projection>
Summary summarize(std::size_t shape_index, Projection projection) {
    std::array<float, kWorldSize> values{};
    for (int rank = 0; rank < kWorldSize; ++rank)
        values[rank] = projection(g_timings[shape_index][rank]);
    return {*std::min_element(values.begin(), values.end()),
            std::accumulate(values.begin(), values.end(), 0.0f) / kWorldSize,
            *std::max_element(values.begin(), values.end())};
}

double gb_per_second(std::size_t bytes, float milliseconds) {
    return static_cast<double>(bytes) / (static_cast<double>(milliseconds) * 1.0e6);
}

void print_results() {
    std::printf("\nVMM peer, multicast, and QuACK-style memcpy benchmark\n");
    std::printf("K=%d, BF16, 8 multicast ranks, copy bytes = "
                "(M * K * sizeof(BF16)) / 8\n",
                kK);
    std::printf("timing: %d warmup + %d measured iterations per M; "
                "correctness: destination validation PASS\n"
                "QuACK rows time data pushes only (local staging and flags excluded).\n\n",
                kWarmupIterations,
                kIterations);
    std::printf("%7s %10s  %-31s %12s %14s %17s\n",
                "M",
                "slice MiB",
                "method",
                "latency (ms)",
                "input GB/s",
                "delivered GB/s");

    for (std::size_t shape_index = 0; shape_index < kMValues.size(); ++shape_index) {
        const int m = kMValues[shape_index];
        const std::size_t bytes_per_shard = shard_bytes(m);
        const float peer_ms = g_timings[shape_index][0].peer_pull_ms;
        const float single_mc_ms = g_timings[shape_index][0].single_mc_push_ms;
        const float peer_push_ms = g_timings[shape_index][1].single_peer_push_ms;
        const Summary concurrent_mc = summarize(
            shape_index, [](const Timings& timing) { return timing.concurrent_mc_push_ms; });
        const Summary quack_ag = summarize(
            shape_index, [](const Timings& timing) { return timing.quack_ag_push_ms; });

        const double slice_mib = static_cast<double>(bytes_per_shard) / (1 << 20);
        const std::size_t single_mc_delivered = bytes_per_shard * kWorldSize;
        const std::size_t concurrent_input = bytes_per_shard * kWorldSize;
        const std::size_t concurrent_delivered =
            bytes_per_shard * kWorldSize * kWorldSize;
        const std::size_t quack_remote_bytes =
            bytes_per_shard * kWorldSize * (kWorldSize - 1);

        std::printf("%7d %10.2f  %-31s %12.3f %14.1f %17.1f\n",
                    m,
                    slice_mib,
                    "rank 0 pulls from rank 1",
                    peer_ms,
                    gb_per_second(bytes_per_shard, peer_ms),
                    gb_per_second(bytes_per_shard, peer_ms));
        std::printf("%7s %10s  %-31s %12.3f %14.1f %17.1f\n",
                    "",
                    "",
                    "rank 0 multicast push",
                    single_mc_ms,
                    gb_per_second(bytes_per_shard, single_mc_ms),
                    gb_per_second(single_mc_delivered, single_mc_ms));
        std::printf("%7s %10s  %-31s %12.3f %14.1f %17.1f\n",
                    "",
                    "",
                    "8 concurrent multicast pushes",
                    concurrent_mc.maximum,
                    gb_per_second(concurrent_input, concurrent_mc.maximum),
                    gb_per_second(concurrent_delivered, concurrent_mc.maximum));
        std::printf("%7s %10s  %-31s %12.3f %14.1f %17.1f\n",
                    "",
                    "",
                    "rank 1 pushes into rank 0",
                    peer_push_ms,
                    gb_per_second(bytes_per_shard, peer_push_ms),
                    gb_per_second(bytes_per_shard, peer_push_ms));
        std::printf("%7s %10s  %-31s %12.3f %14.1f %17.1f\n",
                    "",
                    "",
                    "QuACK AG (8 x 7 peer pushes)",
                    quack_ag.maximum,
                    gb_per_second(quack_remote_bytes, quack_ag.maximum),
                    gb_per_second(quack_remote_bytes, quack_ag.maximum));
        std::printf("%20s latency ratios: single-MC/pull %.2fx, peer-push/pull %.2fx, "
                    "8x-MC/single-MC %.2fx, QuACK-AG/8x-MC %.2fx\n",
                    "",
                    single_mc_ms / peer_ms,
                    peer_push_ms / peer_ms,
                    concurrent_mc.maximum / single_mc_ms,
                    quack_ag.maximum / concurrent_mc.maximum);
        std::printf("%20s per-rank min/avg/max ms: 8x-MC %.3f/%.3f/%.3f, "
                    "QuACK-AG %.3f/%.3f/%.3f\n",
                    "",
                    concurrent_mc.minimum,
                    concurrent_mc.average,
                    concurrent_mc.maximum,
                    quack_ag.minimum,
                    quack_ag.average,
                    quack_ag.maximum);
    }
}

}  // namespace

RankState g_rs[8];

int main() {
    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", -1))
        return 1;
    if (device_count < kWorldSize) {
        std::fprintf(stderr,
                     "This benchmark requires %d GPUs, but only %d are visible.\n",
                     kWorldSize,
                     device_count);
        return 1;
    }

    std::vector<std::thread> threads;
    threads.reserve(kWorldSize);
    for (int rank = 0; rank < kWorldSize; ++rank)
        threads.emplace_back(benchmark_rank, rank);
    for (std::thread& thread : threads)
        thread.join();

    if (g_failed.load(std::memory_order_relaxed))
        return 1;

    print_results();
    return 0;
}
