#pragma once

#include "external.hpp"

#include "compute/buffer.hpp"
#include "compute/generic.hpp"
#include "compute/executor.hpp"

template <typename T, unsigned int nDim>
class Matrix;

template <typename T>
using Matrix1D = Matrix<T, 1>;

template <typename T>
using Matrix2D = Matrix<T, 2>;

template <typename T>
using Matrix3D = Matrix<T, 3>;

template <typename T, unsigned int nDim>
class Matrix {

	private:

		Buffer<T> buffer;
		unsigned int dimensions[nDim];

		template <typename... Args>
		__host__ __device__ inline const auto getIndex(Args... args) const {
			const int argsArr[] = { args... };
			constexpr int argSize = sizeof...(Args);
			constexpr int argNDim = (argSize > nDim) ? nDim : argSize;

			#if !defined NDEBUG && !defined __CUDA_ARCH__
			for (int i = 0; i < argNDim; i++) {
				if (argsArr[i] >= dimensions[i]) {
					throw std::out_of_range("Index out of range");
				}
			}
			#endif

			int index = 0;
			int multi = 1;

			constexpr int argDiff = nDim - argNDim;
			if constexpr (argNDim < nDim) {
				for (int i = nDim - 1; i >= argNDim; i--) {
					multi *= dimensions[i];
				}
			}

			for (int i = argNDim - 1; i >= 0; i--) {
				index += argsArr[i] * multi;
				multi *= dimensions[i];
			}

			return index;
		}

	public:		

		/**
		* Creation, destruction, copying
		*/

		__host__ Matrix() : buffer() {
			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = 0;
			}
		}

		__host__ Matrix(const std::array<unsigned int, nDim>& dimensions, const std::initializer_list<T>& values = {}) : buffer() {
			int size = 1;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = dimensions[i];
				size *= dimensions[i];
			}

			this->buffer.resize(size);

			if (values.size() > 0) {
				*this = values;
			}
		}

		__host__ __device__ Matrix(const Matrix& other) : buffer(other.buffer) {
			*this = other;
		}

		__host__ __device__ Matrix& operator=(const Matrix& other) {
			this->buffer = other.buffer;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = other.dimensions[i];
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
		__host__ __device__ decltype(auto) operator()(Args... args) {
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

		template <bool transposeA = false>
		GENERIC_KERNEL(MatrixMultiplyKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<T> matA, Matrix2D<T> matB, Matrix2D<T> matC) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= matC.shape(1) || index.y >= matC.shape(0)) {
					return;
				}

				if constexpr (transposeA) {
					unsigned int aRows = matA.shape(0);

					T sum = 0;

					for (unsigned int i = 0; i < aRows; i++) {
						sum += matA(i, index.y) * matB(i, index.x);
					}

					matC(index.y, index.x) = sum;
				}
				else {
					unsigned int aCols = matA.shape(1);

					T sum = 0;

					for (unsigned int i = 0; i < aCols; i++) {
						sum += matA(index.y, i) * matB(i, index.x);
					}

					matC(index.y, index.x) = sum;
				}
			}
		};

		template <typename Exe>
		__host__ static void multiply(Exe& executor, Matrix2D<T>& matA, Matrix2D<T>& matB, Matrix2D<T>& matC, bool transposeA = false) {
			if (transposeA) {
				if (matA.shape(0) != matB.shape(0) || matA.shape(1) != matC.shape(0) || matB.shape(1) != matC.shape(1)) {
					throw std::invalid_argument("Matrix dimensions do not match");
				}

				executor.template execute<MatrixMultiplyKernel<true>>({ matC.shape(1), matC.shape(0) }, matA, matB, matC);
			}
			else {
				if (matA.shape(1) != matB.shape(0) || matA.shape(0) != matC.shape(0) || matB.shape(1) != matC.shape(1)) {
					throw std::invalid_argument("Matrix dimensions do not match");
				}

				executor.template execute<MatrixMultiplyKernel<false>>({ matC.shape(1), matC.shape(0) }, matA, matB, matC);
			}
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

		__host__ friend std::ostream& operator<<(std::ostream& os, Matrix& matrix) {
			// print dimensions
			os << "(";
			for (unsigned int i = 0; i < nDim; i++) {
				os << matrix.dimensions[i];
				if (i < nDim - 1) {
					os << ", ";
				}
			}
			os << ") ";

			// print data
			os << "[";
			if (nDim == 1) {
				for (unsigned int i = 0; i < matrix.buffer.size(); i++) {
					os << matrix.buffer[i];
					if (i < matrix.buffer.size() - 1) {
						os << ", ";
					}
				}
			}
			else if (nDim >= 2) {
				for (unsigned int i = 0; i < matrix.buffer.size(); i++) {
					if (i % matrix.dimensions[1] == 0) {
						os << "\n   ";
					}
					os << matrix.buffer[i];
					if (i < matrix.buffer.size() - 1) {
						os << ", ";
					}
				}
			}
			os << "]";

			return os;
		}

		__host__ __device__ unsigned int shape(unsigned int dim) const {
			return dimensions[dim];
		}

		Buffer<T>& getBuffer() {
			return buffer;
		}

};
