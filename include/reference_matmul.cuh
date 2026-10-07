#pragma once

/**
 * @file
 * @brief cuBLAS reference GEMM + tolerance-based comparison for
 * correctness-checking a device GEMM kernel's output.
 */

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

// Reference GEMM via cuBLAS: C[m,n] = sum_k A[m,k] * B[n,k], for a row-major
// A[M,K] and a [N,K] "transposed" B -- matching how ag_gemm_warp_specialized
// stores its weight operand (fused_globals::B_tile: rows are the N-chunk,
// cols are the K-chunk, read back with transpose::T).
//
// A_host/B_host are the same bf16-rounded floats fill_random returned (so
// this reference sees exactly what's on device, not pre-rounding randoms).
// Inputs are uploaded as bf16 and multiplied with fp32 accumulation via
// cublasGemmEx -- fast enough to check the *full* M x N output rather than
// a small corner, unlike a naive host loop (which would be ~1e11 MACs at
// this kernel's production sizes).
//
// cuBLAS is column-major; row-major A[M,K] @ B[N,K]^T is computed by
// reinterpreting each row-major buffer as its column-major transpose and
// swapping operands: C_col(N,M) = op(B_buf)_col(N,K) @ op(A_buf)_col(K,M),
// which is exactly C_row(M,N) once the output buffer is read back row-major.
inline std::vector<float> reference_matmul_cublas(const std::vector<float>& A_host,
                                                   int M,
                                                   int K,
                                                   const std::vector<float>& B_host,
                                                   int N) {
    std::vector<__nv_bfloat16> A_bf16(A_host.size());
    for (size_t i = 0; i < A_host.size(); ++i)
        A_bf16[i] = __float2bfloat16(A_host[i]);
    std::vector<__nv_bfloat16> B_bf16(B_host.size());
    for (size_t i = 0; i < B_host.size(); ++i)
        B_bf16[i] = __float2bfloat16(B_host[i]);

    __nv_bfloat16* A_dev = nullptr;
    __nv_bfloat16* B_dev = nullptr;
    float* C_dev = nullptr;
    cudaMalloc(&A_dev, A_bf16.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&B_dev, B_bf16.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&C_dev, static_cast<size_t>(M) * N * sizeof(float));
    cudaMemcpy(A_dev, A_bf16.data(), A_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_bf16.data(), B_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(handle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           N,
                                           M,
                                           K,
                                           &alpha,
                                           B_dev,
                                           CUDA_R_16BF,
                                           K,
                                           A_dev,
                                           CUDA_R_16BF,
                                           K,
                                           &beta,
                                           C_dev,
                                           CUDA_R_32F,
                                           N,
                                           CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "reference_matmul_cublas: cublasGemmEx failed (status %d)\n", (int)st);
    }

    std::vector<float> C_host(static_cast<size_t>(M) * N);
    cudaMemcpy(C_host.data(), C_dev, C_host.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    return C_host;
}

// Element-wise compares two row-major [rows x cols] buffers within
// atol + rtol*|expected|. Defaults (atol=1e-1, rtol=2e-2) are a starting
// point for bf16 inputs with fp32 accumulation -- bf16 has ~3 significant
// decimal digits, so expect per-element error on that order; tighten once
// you've confirmed the kernel is correct, or loosen if it accumulates in
// bf16 instead of fp32. Prints the first few mismatches on failure.
inline bool check_close(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        int rows,
                        int cols,
                        float atol = 1e-1f,
                        float rtol = 2e-2f,
                        const char* label = "C") {
    const size_t expected_count = static_cast<size_t>(rows) * cols;
    if (actual.size() != expected_count || expected.size() != expected_count) {
        std::fprintf(stderr,
                     "%s: size mismatch (actual=%zu, expected=%zu, rows*cols=%zu)\n",
                     label,
                     actual.size(),
                     expected.size(),
                     expected_count);
        return false;
    }

    bool ok = true;
    int mismatches = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const size_t idx = static_cast<size_t>(r) * cols + c;
            const float a = actual[idx];
            const float e = expected[idx];
            const float tol = atol + rtol * std::fabs(e);
            if (std::fabs(a - e) > tol) {
                if (mismatches < 5) {
                    std::fprintf(stderr,
                                 "%s mismatch at (row %d, col %d): got %.4f, expected %.4f (tol %.4f)\n",
                                 label,
                                 r,
                                 c,
                                 a,
                                 e,
                                 tol);
                }
                ++mismatches;
                ok = false;
            }
        }
    }
    if (!ok) {
        std::fprintf(stderr, "%s: %d/%d elements out of tolerance\n", label, mismatches, rows * cols);
    }
    return ok;
}
