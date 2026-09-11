#pragma once

#include <cuda.h>

#include <mutex>

constexpr int kDmwChannels = 2;
constexpr size_t kSlab = (512ull << 20);
constexpr size_t kFlagOff = 320ull << 20;

struct RankState {
    CUdeviceptr uc = 0, mc = 0;
    unsigned ag_calls = 0, rs_calls = 0;
    bool ready = false;   // SP_MM twins usable
    bool mapped = false;  // slab mapped (either arm)
    bool dmw = false;
    unsigned* dmw_cnt[kDmwChannels] = {nullptr, nullptr};
    unsigned* lat_cnt = nullptr;
};
extern RankState g_rs[8];

int minfer_spmm_init(int rank, int world, int device, size_t data_bytes);