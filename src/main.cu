
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"

GENERIC_KERNEL(GenericAddKernel) {

    GENERIC_KERNEL_ENTRY(Matrix2D<float> c, Matrix1D<float> a, Matrix1D<float> b) {
        int i = getThreadIdx().x;
		c(0, i) = gfma(b(i), 1, a(i));
		c(1, i) = gfma(b(i), 2, a(i));
    }

};

int main() {

	const unsigned int size = 5;

	Matrix1D<float> dev_a({ size });
	dev_a = { 1, 2, 3, 4, 5 };
	Matrix1D<float> dev_b({ size });
	dev_b = { 10, 20, 30, 40, 50 };
	Matrix2D<float> dev_c({ 2, size });

	CUDAExecutor executor;

	executor.execute<GenericAddKernel>(size, dev_c, dev_a, dev_b);

	executor.synchronize(dev_c);

	std::cout << "Matrix A: " << dev_a << "\n\n";
	std::cout << "Matrix B: " << dev_b << "\n\n";
	std::cout << "Matrix C: " << dev_c << "\n\n";

    return 0;
}

