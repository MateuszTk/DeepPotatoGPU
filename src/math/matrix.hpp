#pragma once

#include "external.hpp"

#include "compute/buffer.hpp"
#include "compute/generic.hpp"
#include "compute/executor.hpp"
#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

template <typename T, unsigned int nDim>
class Matrix;

template <typename T>
using Matrix1D = Matrix<T, 1>;

template <typename T>
using Matrix2D = Matrix<T, 2>;

template <typename T>
using Matrix3D = Matrix<T, 3>;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <typename T, unsigned int nDim>
class Matrix {

	private:

		Buffer<T> buffer;
		unsigned int dimensions[nDim];
		unsigned int dataDimensions[nDim];

		template <typename... Args>
		__host__ __device__ inline const auto getIndex(Args... args) const {
			const int argsArr[] = { args... };
			constexpr int argSize = sizeof...(Args);
			constexpr int argNDim = (argSize > nDim) ? nDim : argSize;

			#if !defined NDEBUG && !defined __CUDA_ARCH__
			for (int i = 0; i < argNDim; i++) {
				// TODO: add non-data dimension checks
				if (argsArr[i] >= dataDimensions[i]) {
					throw std::out_of_range("Index out of range");
				}
			}
			#endif

			int index = 0;
			int multi = 1;

			constexpr int argDiff = nDim - argNDim;
			if constexpr (argNDim < nDim) {
				for (int i = nDim - 1; i >= argNDim; i--) {
					multi *= dataDimensions[i];
				}
			}

			for (int i = argNDim - 1; i >= 0; i--) {
				index += argsArr[i] * multi;
				multi *= dataDimensions[i];
			}

			return index;
		}

		__host__ __device__ static int padding(int value, int multiple) {
			return (value + multiple - 1) / multiple * multiple;
		}

		__host__ __device__ static int getPaddingForDim(int value, int dim) {
			#if !defined NDEBUG && !defined __CUDA_ARCH__
			if (dim >= nDim) {
				throw std::out_of_range("Index out of range");
			}
			#endif
			#if USE_WMMA == 1
			const int paddings[] = { WMMA_M, WMMA_N, WMMA_K };
			return padding(value, paddings[dim]);
			#else
			return value;
			#endif	
		}

	public:		

		using type = T;

		/**
		* Creation, destruction, copying
		*/

		__host__ Matrix() : buffer() {
			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = 0;
				this->dataDimensions[i] = 0;
			}
		}

		__host__ Matrix(const std::array<unsigned int, nDim>& dimensions, const std::initializer_list<T>& values = {}) : buffer() {
			int dataSize = 1;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = dimensions[i];
				this->dataDimensions[i] = getPaddingForDim(dimensions[i], i);
				dataSize *= this->dataDimensions[i];
			}

			this->buffer.resize(dataSize);

			if (values.size() > 0) {
				*this = values;
			}
		}

		__host__ Matrix(Matrix&& other) : buffer(std::move(other.buffer)) {
			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = other.dimensions[i];
				this->dataDimensions[i] = other.dataDimensions[i];
			}
		}

		__host__ __device__ Matrix(const Matrix& other) : buffer(other.buffer) {
			*this = other;
		}

		__host__ __device__ Matrix& operator=(const Matrix& other) {
			this->buffer = other.buffer;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = other.dimensions[i];
				this->dataDimensions[i] = other.dataDimensions[i];
			}

			return *this;
		}

		__host__ Matrix& operator=(const std::initializer_list<T>& values) {
			if (values.size() > buffer.size()) {
				throw std::invalid_argument("Invalid number of elements");
			}

			buffer.store(values.begin(), values.size());

			return *this;
		}

		template <unsigned int nDimOther>
		__host__ Matrix<T, nDimOther> reshape(const std::array<unsigned int, nDimOther>& dimensions) {
			Matrix<T, nDimOther> result(dimensions);

			if (result.buffer.size() != buffer.size()) {
				throw std::invalid_argument("Invalid number of elements");
			}

			result.buffer = buffer;

			for (unsigned int i = 0; i < nDimOther; i++) {
				result.dimensions[i] = dimensions[i];
			}

			return result;
		}

		/**
		* Data access
		*/

		template <typename... Args>
		__host__ __device__ FORCEINLINE decltype(auto) operator()(Args... args) {
			constexpr unsigned int argSize = sizeof...(Args);
			const unsigned int index = getIndex(args...);
			
			if constexpr (argSize >= nDim) {
				return (T&)(buffer[index]);
			}
			else {
				return Matrix<T, nDim - argSize>(*this, index);
			}
		}

		/**
		* Math operations
		*/

		GENERIC_KERNEL(MatrixMultiplyKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<T> matA, Matrix2D<T> matB, Matrix2D<T> matC) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= matC.shape(1) || index.y >= matC.shape(0)) {
					return;
				}

				unsigned int aCols = matA.shape(1);

				T sum = 0;

				for (unsigned int i = 0; i < aCols; i++) {
					sum += matA(index.y, i) * matB(i, index.x);
				}

				matC(index.y, index.x) = sum;
			}
		};

		__host__ static void multiply(CUDAExecutor& executor, Matrix2D<T>& matA, Matrix2D<T>& matB, Matrix2D<T>& matC) {
			if (matA.shape(1) != matB.shape(0) || matA.shape(0) != matC.shape(0) || matB.shape(1) != matC.shape(1)) {
				throw std::invalid_argument("Matrix dimensions do not match");
			}

			executor.execute<MatrixMultiplyKernel>({ matC.shape(1), matC.shape(0) }, matA, matB, matC);
		}

		GENERIC_KERNEL(MatrixMultiplyKernelWMMA) {
			__device__ void operator()(Matrix2D<half> matA, Matrix2D<half> matB, Matrix2D<float> matC) {
				int tileM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
				int tileN = blockIdx.y;

				int N = matB.dataShape(1);
				int K = matA.dataShape(1);

				wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
				wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
				wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

				wmma::fill_fragment(c_frag, 0.0f);

				for (int tileK = 0; tileK < K / WMMA_K; tileK++) {
					const half* tile_a = &matA(tileM * WMMA_M, tileK * WMMA_K);
					const half* tile_b = &matB(tileK * WMMA_K, tileN * WMMA_N);

					wmma::load_matrix_sync(a_frag, tile_a, K);
					wmma::load_matrix_sync(b_frag, tile_b, N);

					wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
				}

				float* c_tile = &matC(tileM * WMMA_M, tileN * WMMA_N);
				wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
			}
		};

		__host__ static LaunchParams getWMMALaunchSize(int matrixM, int matrixN) {
			matrixM = getPaddingForDim(matrixM, 0);
			matrixN = getPaddingForDim(matrixN, 1);

			// TODO: move this to a more appropriate place to avoid multiple calls
			int warpSize = 0;
			cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);

			LaunchParams launchParams;
			launchParams.blocks = { (uint32_t)matrixM / WMMA_M, (uint32_t)matrixN / WMMA_N, 1 };
			launchParams.threads = { (uint32_t)warpSize, 1, 1 }; // TODO: Launch more warps per block

			return launchParams;
		}

		__host__ static void multiplyWMMA(CUDAExecutor& executor, Matrix2D<half>& matA, Matrix2D<half>& matB, Matrix2D<float>& matC) {
			if (matA.shape(1) != matB.shape(0) || matA.shape(0) != matC.shape(0) || matB.shape(1) != matC.shape(1)) {
				throw std::invalid_argument("Matrix dimensions do not match");
			}
			if (matA.dataShape(1) % 16 != 0 || matB.dataShape(0) % 16 != 0 || matC.dataShape(1) % 16 != 0 || matC.dataShape(0) % 16 != 0) {
				throw std::invalid_argument("Matrix dimensions must be multiples of 16 for WMMA");
			}

			const int matrixM = matA.dataShape(0);
			const int matrixN = matB.dataShape(1);
			const int matrixK = matA.dataShape(1);

			LaunchParams launchParams = getWMMALaunchSize(matrixM, matrixN);

			executor.executeParams<MatrixMultiplyKernelWMMA>(launchParams,
				matA, matB, matC
			);
		}

		GENERIC_KERNEL(MatrixScalarMultiplyKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<T> matA, T scalar, Matrix2D<T> matB) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= matB.shape(1) || index.y >= matB.shape(0)) {
					return;
				}

				matB(index.y, index.x) = matA(index.y, index.x) * scalar;
			}
		};

		template <typename Exe>
		__host__ static void multiply(Exe& executor, Matrix2D<T>& matA, T scalar, Matrix2D<T>& matB) {
			executor.template execute<MatrixScalarMultiplyKernel>({ matB.shape(1), matB.shape(0) }, matA, scalar, matB);
		}

		GENERIC_KERNEL(MatrixAddKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<T> matA, Matrix2D<T> matB, Matrix2D<T> matC) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= matC.shape(1) || index.y >= matC.shape(0)) {
					return;
				}

				matC(index.y, index.x) = matA(index.y, index.x) + matB(index.y, index.x);
			}
		};

		template <typename Exe>
		__host__ static void add(Exe& executor, Matrix2D<T>& matA, Matrix2D<T>& matB, Matrix2D<T>& matC) {

			for (unsigned int i = 0; i < nDim; i++) {
				if (matA.shape(i) != matB.shape(i) || matA.shape(i) != matC.shape(i)) {
					throw std::invalid_argument("Matrix dimensions do not match");
				}
			}

			executor.template execute<MatrixAddKernel>({ matC.shape(1), matC.shape(0) }, matA, matB, matC);
		}

		GENERIC_KERNEL(MatrixSubtractKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<T> matA, Matrix2D<T> matB, Matrix2D<T> matC) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= matC.shape(1) || index.y >= matC.shape(0)) {
					return;
				}

				matC(index.y, index.x) = matA(index.y, index.x) - matB(index.y, index.x);
			}
		};

		template <typename Exe>
		__host__ static void subtract(Exe& executor, Matrix2D<T>& matA, Matrix2D<T>& matB, Matrix2D<T>& matC) {
			if (matA.shape(0) != matB.shape(0) || matA.shape(1) != matB.shape(1) || matA.shape(0) != matC.shape(0) || matA.shape(1) != matC.shape(1)) {
				throw std::invalid_argument("Matrix dimensions do not match");
			}

			executor.template execute<MatrixSubtractKernel>({ matC.shape(1), matC.shape(0) }, matA, matB, matC);
		}

		/**
		* Other
		*/

		__host__ void fill(const T& value) {
			for (unsigned int i = 0; i < buffer.size(); i++) {
				buffer[i] = value;
			}
		}

		void print(std::ostream& os, bool full = false) {
			// print dimensions
			os << "(";
			for (unsigned int i = 0; i < nDim; i++) {
				os << dimensions[i];
				if (i < nDim - 1) {
					os << ", ";
				}
			}
			os << ") ";

			int printShape[nDim];
			for (unsigned int i = 0; i < nDim; i++) {
				printShape[i] = full ? dataDimensions[i] : dimensions[i];
			}

			// print data
			os << "[";
			if constexpr (nDim == 1) {
				for (unsigned int i = 0; i < printShape[0]; i++) {
					os << static_cast<float>(operator()(i));
					if (i < printShape[0] - 1) {
						os << ", ";
					}
				}
			}
			else if (nDim >= 2) {
				for (unsigned int y = 0; y < printShape[0]; y++) {
					for (unsigned int x = 0; x < printShape[1]; x++) {
						os << static_cast<float>(operator()(y, x));
						if (x < printShape[1] - 1) {
							os << ", ";
						}
					}
					if (y < printShape[0] - 1) {
						os << "\n   ";
					}
				}
			}
			os << "]";
		}

		__host__ friend std::ostream& operator<<(std::ostream& os, Matrix& matrix) {
			matrix.print(os);
			return os;
		}

		__host__ __device__ unsigned int shape(unsigned int dim) const {
			return dimensions[dim];
		}

		__host__ __device__ unsigned int dataShape(unsigned int dim) const {
			return dataDimensions[dim];
		}

		Buffer<T>& getBuffer() {
			return buffer;
		}

};
