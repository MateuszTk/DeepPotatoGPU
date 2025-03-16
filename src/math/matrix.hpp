#pragma once

#include <initializer_list>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "compute/buffer.hpp"

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

		__host__ Matrix() : buffer() {
			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = 0;
			}
		}

		__host__ Matrix(const std::array<unsigned int, nDim>& dimensions) : buffer() {
			int size = 1;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = dimensions[i];
				size *= dimensions[i];
			}

			this->buffer.resize(size);
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
			if (values.size() != buffer.size()) {
				throw std::invalid_argument("Invalid number of elements");
			}

			buffer.store(values.begin(), values.size());

			return *this;
		}

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

		Buffer<T>& getBuffer() {
			return buffer;
		}

};

template <typename T>
using Matrix1D = Matrix<T, 1>;

template <typename T>
using Matrix2D = Matrix<T, 2>;

template <typename T>
using Matrix3D = Matrix<T, 3>;

/*template <typename T, unsigned int nDim>
class Matrix {

	private:

		std::shared_ptr<Buffer<T>> data;
		unsigned int dataOffset;
		unsigned int size;
		unsigned int dimensions[nDim];
		bool isSubMatrix;

		template <typename... Args>
		inline const auto getIndex(Args... args) const {
			const int argsArr[] = { args... };
			constexpr int argSize = sizeof...(Args);
			constexpr int argNDim = (argSize > nDim) ? nDim : argSize;

			#ifndef NDEBUG
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

		Matrix() : isSubMatrix(false), data(std::make_shared<Buffer<T>>()) {
			this->size = 0;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = 0;
			}
		}

		Matrix(const std::array<unsigned int, nDim>& dimensions) : isSubMatrix(false) {
			this->size = 1;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = dimensions[i];
				this->size *= dimensions[i];
			}

			this->data.resize(size);

			this->size = size;
		}

		Matrix(const Matrix<T, nDim + 1>& master, unsigned int index) : isSubMatrix(true) {
			this->size = 1;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = master.shape(i + 1);
				this->size *= master.shape(i + 1);
			}

			if (this->size > 0) {
				this->data = master.getData() + index;
			}
			else {
				this->data = nullptr;
			}
		}

		Matrix(const Matrix& other) {
			this->data = nullptr;
			*this = other;
		}

		Matrix(Matrix&& other) noexcept {
			this->data = other.data;
			this->size = other.size;
			this->isSubMatrix = other.isSubMatrix;

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = other.dimensions[i];
			}

			other.data = nullptr;
			other.size = 0;
			other.isSubMatrix = false;
		}

		~Matrix() {
			if (!isSubMatrix) {
				delete[] data;
			}
		}

		// TODO: Do not allow to resize submatrices
		Matrix& operator=(const Matrix& other) {
			if (this->data != nullptr && !this->isSubMatrix) {
				delete[] this->data;
			}

			this->size = other.size;
			this->isSubMatrix = other.isSubMatrix;

			if (this->size > 0) {
				if (!this->isSubMatrix) {
					this->data = new T[size];
				}
				else {
					this->data = other.data;
				}
			}
			else {
				this->data = nullptr;
			}

			for (unsigned int i = 0; i < nDim; i++) {
				this->dimensions[i] = other.dimensions[i];
			}

			memcpy(this->data, other.data, size * sizeof(T));

			return *this;
		}

		Matrix& operator=(const std::initializer_list<T>& values) {
			if (values.size() != size) {
				throw std::invalid_argument("Invalid number of elements");
			}

			memcpy(this->data, values.begin(), size * sizeof(T));

			return *this;
		}

		template <typename... Args>
		decltype(auto) operator()(Args... args) const {
			constexpr unsigned int argSize = sizeof...(Args);
			const int index = getIndex(args...);

			if constexpr (argSize >= nDim) {
				return (T&)(data[index]);
			}
			else {
				return Matrix<T, nDim - argSize>(*this, index);
			}
		}

		const T* begin() const {
			return data;
		}

		const T* end() const {
			return data + size;
		}

		T* getData() const {
			return data;
		}

		unsigned int shape(unsigned int dim) const {
			return dimensions[dim];
		}

		void fill(T value) {
			for (unsigned int i = 0; i < size; i++) {
				data[i] = value;
			}
		}

		template <typename... Args>
		T* at(Args... args) const {
			return data + getIndex(args...);
		}

		static Matrix identity(const unsigned int size) {
			Matrix<T, 2> result({ size, size });
			result.fill(0);

			for (unsigned int i = 0; i < size; i++) {
				result(i, i) = 1;
			}

			return result;
		}

		friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
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
				for (unsigned int i = 0; i < matrix.size; i++) {
					os << matrix.data[i];
					if (i < matrix.size - 1) {
						os << ", ";
					}
				}
			}
			else if (nDim >= 2) {
				for (unsigned int i = 0; i < matrix.size; i++) {
					if (i % matrix.dimensions[1] == 0) {
						os << "\n   ";
					}
					os << matrix.data[i];
					if (i < matrix.size - 1) {
						os << ", ";
					}
				}
			}
			os << "]";

			return os;
		}

};

template <typename T>
using Matrix1D = Matrix<T, 1>;

template <typename T>
using Matrix2D = Matrix<T, 2>;

template <typename T>
using Matrix3D = Matrix<T, 3>;

void matrixTest() {
	Matrix2D<float> a({ 4, 3 });
	a = {
		1, 0, 1,
		2, 1, 1,
		0, 1, 1,
		1, 1, 2
	};
	std::cout << "Matrix A: " << a << "\n\n";

	Matrix2D<float> b({ 3, 3 });
	b = {
		1, 2, 1,
		2, 3, 1,
		4, 2, 2
	};
	std::cout << "Matrix B: " << b << "\n\n";

	//Matrix2D<float> multiplied = a * b;
	//std::cout << "Matrix A * B: " << multiplied << "\n\n";

	std::cout << "a(1): " << a(1) << "\n\n";

	Matrix3D<float> c({ 2, 2, 2 });
	c = {
		1, 2,
		3, 4,

		5, 6,
		7, 8
	};
	std::cout << "Matrix C: " << c << "\n\n";

	Matrix2D<float> a1 = c(1);
	Matrix1D<float> a10 = a1(0);
	std::cout << a10 << "\n\n";

	std::cout << c(1)(0) << "\n\n";
}
*/