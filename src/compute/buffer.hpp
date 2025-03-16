#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdexcept>
#include <concepts>
#include <iostream>

template <typename T>
concept IsBuffer = requires(T a) {
	{ a.isBuffer() } -> std::same_as<bool>;
};

template <typename T>
concept ContainsBuffer = requires(T a) {
	{ a.getBuffer() } -> IsBuffer;
};

// TODO: Read-only buffers. Does passing const T* make any difference? Performance?

template <typename T>
class Buffer {

	private:

		bool dirty = false;
		T* dataHost = nullptr;
		T* dataDevice = nullptr;
		unsigned int count = 0;
		bool isCopy = false;

		__host__ T* getDataDevice() {
			if (dataDevice == nullptr) {
				if (cudaMalloc(&dataDevice, count * sizeof(T)) != cudaSuccess) {
					throw std::runtime_error("Failed to allocate device memory");
				}
			}

			return dataDevice;
		}

		__host__ void copyToDevice() {
			if (dataDevice == nullptr) {
				if (cudaMalloc(&dataDevice, count * sizeof(T)) != cudaSuccess) {
					throw std::runtime_error("Failed to allocate device memory");
				}
			}

			if (dirty) {
				if (cudaMemcpy(dataDevice, dataHost, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
					throw std::runtime_error("Failed to copy data to device");
				}
			}
		}

		__host__ T* getDataHost() {
			return dataHost;
		}

		__host__ void copyToHost() {
			if (cudaMemcpy(dataHost, dataDevice, count * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
				throw std::runtime_error("Failed to copy data to host");
			}
		}

	public:

		bool isBuffer() {
			return true;
		}

		__host__ Buffer(unsigned int count = 0) {
			this->count = count;

			if (count > 0) {
				this->dataHost = new T[count];
			}
			else {
				this->dataHost = nullptr;
			}
		}

		__host__ __device__ Buffer(const Buffer& other) {
			*this = other;
		}

		__host__ __device__ ~Buffer() {
			#ifndef __CUDA_ARCH__
				if (!isCopy) {
					if (dataDevice != nullptr) {
						cudaFree(dataDevice);
					}
					if (dataHost != nullptr) {
						delete[] dataHost;
					}
				}
			#endif
		}

		__host__ __device__ Buffer& operator=(const Buffer& other) {
			this->count = other.count;
			this->dirty = other.dirty;
			this->dataDevice = other.dataDevice;
			this->dataHost = other.dataHost;
			this->isCopy = true;
			return *this;
		}

		__host__ void resize(unsigned int count) {
			if (dataHost != nullptr) {
				delete[] dataHost;
			}
			if (dataDevice != nullptr) {
				cudaFree(dataDevice);
			}

			this->count = count;

			if (count > 0) {
				this->dataHost = new T[count];
			}
			else {
				this->dataHost = nullptr;
			}
		}

		__host__ void store(const T* data, unsigned int count) {
			memcpy(this->dataHost, data, count * sizeof(T));
			dirty = true;
		}

		__host__ void load(T* data, unsigned int count) {
			if (dataDevice != nullptr) {
				copyToHost();
			}

			memcpy(data, dataHost, count * sizeof(T));
		}

		__host__ __device__ T& operator[](unsigned int index) {
			#ifdef __CUDA_ARCH__
				return dataDevice[index];
			#else
				this->dirty = true;
				return dataHost[index];
			#endif
		}

		__host__ __device__ const T* data() {
			#ifdef __CUDA_ARCH__
				return dataDevice;
			#else
				if (dataDevice != nullptr) {
					copyToHost();
				}

				return dataHost;
			#endif
		}

		__host__ __device__ unsigned int size() const {
			return count;
		}

		friend class Executor;
		friend class CPUExecutor;
		friend class CUDAExecutor;

};
