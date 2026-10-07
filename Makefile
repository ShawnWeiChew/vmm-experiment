main : vmm/main.cpp vmm/multimem.cu mKernel/src/ag_gemm_warp_specialized.cu
	nvcc -gencode arch=compute_103a,code=sm_103a -std=c++20 -x cu \
	    -DKITTENS_SM10X -DKITTENS_BLACKWELL -DMKERNEL_TCGEN05 \
	    --expt-relaxed-constexpr --extended-lambda \
	    -DINTRA_NUM_DEVICES=8 \
	    vmm/main.cpp vmm/multimem.cu mKernel/src/ag_gemm_warp_specialized.cu \
	    -Iinclude -ImKernel/include -lcuda -lcublas -o main

test_vmm_multicast_speed : vmm/test_vmm_multicast_speed.cu vmm/multimem.cu
	nvcc -gencode arch=compute_103a,code=sm_103a -std=c++20 -x cu \
	    vmm/test_vmm_multicast_speed.cu vmm/multimem.cu \
	    -Iinclude -lcuda -o test_vmm_multicast_speed
