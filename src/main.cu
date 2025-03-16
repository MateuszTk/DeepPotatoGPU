
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"

int main() {

	const unsigned int size = 5;

	Matrix2D<float> matA({ 4, 3 });
	matA = {
		1, 0, 1,
		2, 1, 1,
		0, 1, 1,
		1, 1, 2
	};

	Matrix2D<float> matB({ 3, 3 });
	matB = {
		1, 2, 1,
		2, 3, 1,
		4, 2, 2
	};

	Matrix2D<float> matC({ matA.shape(0), matB.shape(1) });
	Matrix2D<float> matC2({ matC.shape(0), matC.shape(1) });

	Matrix2D<float> matD({ 4, 3 });
	matD = {
		10, 0, 0,
		0, 10, 0,
		0, 0, 10,
		0, 0, 10
	};

	std::cout << "Matrix A: " << matA << "\n\n";
	std::cout << "Matrix B: " << matB << "\n\n";

	CUDAExecutor executor;
	Matrix2D<float>::multiply(executor, matA, matB, matC);

	std::cout << "After multiplication of A and B:\n";
	std::cout << "Matrix C: " << matC << "\n\n";

	CPUExecutor cpuExecutor;
	Matrix2D<float>::add(cpuExecutor, matC, matD, matC2);

	std::cout << "After adding matrix D to C:\n";
	std::cout << "Matrix C: " << matC2 << "\n\n";

    return 0;
}

