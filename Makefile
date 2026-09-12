main : vmm/main.cpp vmm/multimem.cu
	nvcc -gencode arch=compute_90,code=sm_90 -std=c++20 -x cu -DKITTENS_SM90 \
	    --expt-relaxed-constexpr --extended-lambda \
	    -DINTRA_NUM_DEVICES=2 \
	    vmm/main.cpp vmm/multimem.cu -Iinclude -ImKernel/include -lcuda -o main