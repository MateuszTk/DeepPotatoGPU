#pragma once

#include "external.hpp"

#if BUFFER_DEBUG_ON
#define BUFFER_LOG(format, ...) printf("[BUFFER] " format, __VA_ARGS__)
#else
#define BUFFER_LOG(...)
#endif

template <typename T>
concept IsBuffer = requires(T a) {
	{ a.isBuffer() } -> std::same_as<bool>;
};

template <typename T>
concept ContainsBuffer = requires(T a) {
	{ a.getBuffer() } -> IsBuffer;
};

enum class Location {
	Host,
	Device
};

enum class BufferDirection {
	Bidirectional,
	HostToDevice,
	DeviceToHost
};

template <typename T>
class Buffer {

	private:

		bool dirty = false;
		Location* location = nullptr;
		T* dataHost = nullptr;
		T* dataDevice = nullptr;
		unsigned int count = 0;
		bool isCopy = false;
		BufferDirection direction = BufferDirection::Bidirectional;

		__host__ void copyToDevice() {
			#ifdef CUDA_AVAILIABLE
				if (dataDevice == nullptr) {
					BUFFER_LOG("Allocating device memory (%d bytes)\n", count * sizeof(T));

					if (cudaMalloc(&dataDevice, count * sizeof(T)) != cudaSuccess) {
						throw std::runtime_error("Failed to allocate device memory");
					}
				}

				if (dirty && direction != BufferDirection::DeviceToHost) {
					BUFFER_LOG("Copying data to device (%d bytes)\n", count * sizeof(T));

					if (cudaMemcpy(dataDevice, dataHost, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
						throw std::runtime_error("Failed to copy data to device");
					}
				}
			#endif
		}

		__host__ void copyToHost() {
			#ifdef CUDA_AVAILIABLE
				if (dataDevice != nullptr && direction != BufferDirection::HostToDevice) {
					BUFFER_LOG("Copying data to host (%d bytes)\n", count * sizeof(T));

					if (cudaMemcpy(dataHost, dataDevice, count * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
						throw std::runtime_error("Failed to copy data to host");
					}
				}
			#endif
		}

		__host__ void transitionLocation(Location dstLocation) {
			#ifdef CUDA_AVAILIABLE
				if (*location == Location::Host && dstLocation == Location::Device) {
					copyToDevice();
					*location = Location::Device;
				}
				else if (*location == Location::Device && dstLocation == Location::Host) {
					copyToHost();
					*location = Location::Host;
				}
			#endif
		}

	public:

		__host__ Buffer(unsigned int count = 0) {
			this->location = new Location(Location::Host);

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
					#ifdef CUDA_AVAILIABLE
						if (dataDevice != nullptr) {
							cudaFree(dataDevice);
						}
					#endif
					if (dataHost != nullptr) {
						delete[] dataHost;
					}

					delete location;
				}
			#endif
		}

		__host__ __device__ Buffer& operator=(const Buffer& other) {
			this->count = other.count;
			this->dirty = other.dirty;
			this->dataDevice = other.dataDevice;
			this->dataHost = other.dataHost;
			this->location = other.location;
			this->isCopy = true;

			return *this;
		}

		__host__ void resize(unsigned int count) {
			if (isCopy) {
				throw std::runtime_error("Cannot resize a copied buffer");
			}

			if (dataHost != nullptr) {
				delete[] dataHost;
			}
			#ifdef CUDA_AVAILIABLE
				if (dataDevice != nullptr) {
					cudaFree(dataDevice);
				}
			#endif

			this->count = count;

			if (count > 0) {
				this->dataHost = new T[count];
			}
			else {
				this->dataHost = nullptr;
			}
		}

		__host__ void store(const T* data, unsigned int count) {
			transitionLocation(Location::Host);
			memcpy(this->dataHost, data, count * sizeof(T));
			dirty = true;
		}

		__host__ void load(T* data, unsigned int count) {
			transitionLocation(Location::Host);
			memcpy(data, dataHost, count * sizeof(T));
		}

		__host__ __device__ T& operator[](unsigned int index) {
			#ifdef __CUDA_ARCH__
				return dataDevice[index];
			#else
				transitionLocation(Location::Host);
				this->dirty = true;
				return dataHost[index];
			#endif
		}

		__host__ __device__ const T* data() {
			#ifdef __CUDA_ARCH__
				return dataDevice;
			#else
				transitionLocation(Location::Host);
				return dataHost;
			#endif
		}

		__host__ __device__ unsigned int size() const {
			return count;
		}

		__host__ bool isBuffer() {
			return true;
		}

		__host__ BufferDirection getDirection() {
			return direction;
		}

		__host__ void setDirection(BufferDirection direction) {
			this->direction = direction;
		}

		friend class Executor;
		friend class CPUExecutor;
		friend class CUDAExecutor;

};
