#pragma once

#include <cstdio>

#define MKERNEL_CUCHECK(cmd)                               \
    do {                                                   \
        CUresult err__ = (cmd);                            \
        if (err__ != CUDA_SUCCESS) {                       \
            const char* err_str__ = nullptr;               \
            cuGetErrorString(err__, &err_str__);           \
            std::fprintf(stderr,                           \
                         "CUDA driver error %s:%d '%s'\n", \
                         __FILE__,                         \
                         __LINE__,                         \
                         err_str__ ? err_str__ : "");      \
            std::exit(EXIT_FAILURE);                       \
        }                                                  \
    } while (0)

#define DKF(x)                                                                   \
    do {                                                                         \
        CUresult r_ = (x);                                                       \
        if (r_ != CUDA_SUCCESS) {                                                \
            const char* s_ = "?";                                                \
            cuGetErrorString(r_, &s_);                                           \
            std::fprintf(stderr, "mInfer SP_MM: %s -> %s (disabled)\n", #x, s_); \
            return false;                                                        \
        }                                                                        \
    } while (0)
#define CKF(x)                                                                              \
    do {                                                                                    \
        cudaError_t e_ = (x);                                                               \
        if (e_ != cudaSuccess) {                                                            \
            std::fprintf(                                                                   \
                stderr, "mInfer SP_MM: %s -> %s (disabled)\n", #x, cudaGetErrorString(e_)); \
            return false;                                                                   \
        }                                                                                   \
    } while (0)