#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdexcept>
#include <concepts>
#include <iostream>

// TODO: Read-only buffers. Does passing const T* make any difference? Performance?

template <typename T>
class Buffer {

	private:

		bool dirty = false;
		T* dataHost = nullptr;
		T* dataDevice = nullptr;
		unsigned int count = 0;

		T* getDataDevice() {
			if (dataDevice == nullptr) {
				if (cudaMalloc(&dataDevice, count * sizeof(T)) != cudaSuccess) {
					throw std::runtime_error("Failed to allocate device memory");
				}
			}

			return dataDevice;
		}

		void copyToDevice() {
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

		T* getDataHost() {
			return dataHost;
		}

		void copyToHost() {
			if (cudaMemcpy(dataHost, dataDevice, count * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
				throw std::runtime_error("Failed to copy data to host");
			}
		}

	public:

		Buffer(unsigned int count) {
			this->count = count;
			this->dataHost = new T[count];
		}

		~Buffer() {
			if (dataDevice != nullptr) {
				cudaFree(dataDevice);
			}

			delete[] dataHost;
		}

		void store(const T* data, unsigned int count) {
			memcpy(this->dataHost, data, count * sizeof(T));
			dirty = true;
		}

		void load(T* data, unsigned int count) {
			if (dataDevice != nullptr) {
				copyToHost();
			}
			memcpy(data, dataHost, count * sizeof(T));
		}

		const T* getData() {
			if (dataDevice != nullptr) {
				copyToHost();
			}

			return dataHost;
		}

		friend class Executor;
		friend class CPUExecutor;
		friend class CUDAExecutor;

};

template <typename T>
concept IsBuffer = requires { typename Buffer<T>; };
