
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

GENERIC_KERNEL(GenericAddKernel) {

    GENERIC_KERNEL_ENTRY(int* c, const int* a, const int* b) {
        int i = getThreadIdx().x;
		c[i] = gfma(b[i], 2, a[i]);
    }

};

void add(int* c, const int* a, const int* b, unsigned int size) {

	Buffer<int> dev_a(size);
	Buffer<int> dev_b(size);
	Buffer<int> dev_c(size);

	dev_a.store(a, size);
	dev_b.store(b, size);

	CUDAExecutor executor;

	executor.execute<GenericAddKernel>(size, dev_c, dev_a, dev_b);

	executor.synchronize(dev_c);
	dev_c.load(c, size);
}

int main() {

    const int arraySize = 5;
    int a[arraySize] = { 1, 2, 3, 4, 5 };
    int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	add(c, a, b, arraySize);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    return 0;
}

