main : vmm/main.cpp vmm/multimem.cu
	nvcc -gencode arch=compute_90,code=sm_90 -std=c++20 -x cu vmm/main.cpp vmm/multimem.cu -Iinclude -lcuda -o main