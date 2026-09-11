#include <cstdio>
#include <thread>
#include <vector>

#include "../include/multimem.cuh"

constexpr int num_threads = 2;

RankState g_rs[8];

void thread_func(int device_num) {
    if (!minfer_spmm_init(device_num, num_threads, device_num, 10)) {
        std::printf("Init failed\n");
    }
}

int main() {
    // spawns 2 threads within the same process that are linked to different GPUs
    std::vector<std::thread> ths;

    for (int i = 0; i < num_threads; i++) {
        ths.emplace_back(thread_func, i);
    }

    for (auto& th : ths) {
        th.join();
    }
}