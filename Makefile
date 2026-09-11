main : vmm/main.cpp vmm/multimem.cu
	nvcc vmm/main.cpp vmm/multimem.cu -Iinclude -lcuda -o main