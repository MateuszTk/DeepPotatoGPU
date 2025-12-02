#include "external.hpp"

#include "math/matrix.hpp"
#include "compute/cuda.hpp"
#include "timer.hpp"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

int main() {
	//test();

	const int TILES_M = 64;
	const int TILES_N = 64;
	const int TILES_K = 64;
	const int MATRIX_M = WMMA_M * TILES_M;
	const int MATRIX_N = WMMA_N * TILES_N;
	const int MATRIX_K = WMMA_K * TILES_K;

	Matrix2D<half> a({ MATRIX_M, MATRIX_K });
	Matrix2D<half> b({ MATRIX_K, MATRIX_N });
	Matrix2D<float> af({ MATRIX_M, MATRIX_K });
	Matrix2D<float> bf({ MATRIX_K, MATRIX_N });
	Matrix2D<float> c_mma({ MATRIX_M, MATRIX_N });
	Matrix2D<float> c_ref({ MATRIX_M, MATRIX_N });

	for (int i = 0; i < a.shape(0); i++) {
		for (int j = 0; j < a.shape(1); j++) {
			af(i, j) = (float)(rand() % 3);
			bf(i, j) = (float)(rand() % 3);
			a(i, j) = __float2half(af(i, j));
			b(i, j) = __float2half(bf(i, j));
		}
	}

	CUDAExecutor exec;

	for (int iter = 0; iter < 10; iter++) {
		Matrix2D<float>::multiply(exec, af, bf, c_ref);

		Timer timer1;
		std::cout << "Starting reference multiplication...\n";
		timer1.start();
		for (int loop = 0; loop < 100; loop++) {
			//af(0, 0) += 0;
			//bf(0, 0) += 0;
			//c_ref(0, 0) += 0;
			Matrix2D<float>::multiply(exec, af, bf, c_ref);
			exec.synchronize();
		}
		exec.synchronize();
		timer1.stop();

		Matrix2D<float>::multiplyWMMA(exec, a, b, c_mma);

		std::cout << "Starting WMMA multiplication...\n";
		timer1.start();
		for (int loop = 0; loop < 100; loop++) {
			//a(0, 0) += __float2half(0);
			//b(0, 0) += __float2half(0);
			//c_mma(0, 0) += 0;
			Matrix2D<float>::multiplyWMMA(exec, a, b, c_mma);
			exec.synchronize();
		}
		exec.synchronize();
		timer1.stop();
	}

	/*std::cout << "Input matrix A (reference):\n";
	af.print(std::cout, true);
	std::cout << "Input matrix B (reference):\n";
	bf.print(std::cout, true);
	std::cout << "Result matrix C (reference):\n";
	c_ref.print(std::cout, true);
	std::cout << "Result matrix C (WMMA):\n";
	c_mma.print(std::cout, true);*/

	for (int i = 0; i < a.shape(0); i++) {
		for (int j = 0; j < a.shape(1); j++) {
			float diff = c_ref(i, j) - c_mma(i, j);
			if (fabs(diff) > 1e-2) {
				std::cout << "Mismatch at (" << i << ", " << j << "): expected " << c_ref(i, j)
					<< ", got " << c_mma(i, j) << "\n";
				return -1;
			}
		}
	}

	std::cout << "Result verification PASSED.\n";

	return 0;
}
