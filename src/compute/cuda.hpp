#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "executor.hpp"

template <typename Func, typename... Args>
__global__ void run(Func func, Args... args) {
	func(args...);
}

class CUDAExecutor : public Executor {

	private:

		template <typename T>
		T* extractPointer(Buffer<T>& buffer) {
			buffer.copyToDevice();
			return buffer.getDataDevice();
		}

		template <typename T>
		T extractPointer(T other) {
			return other;
		}

	public:

		CUDAExecutor() {
			if (cudaSetDevice(0) != cudaSuccess) {
				throw std::runtime_error("Failed to set CUDA device");
			}
		}

		virtual ~CUDAExecutor() {
			if (cudaDeviceReset() != cudaSuccess) {
				throw std::runtime_error("Failed to reset CUDA device");
			}
		}

		template <typename Kernel, typename... Args>
		void execute(unsigned int size, Args&... args) {
			Kernel kernel{};

			run << <1, size >> > (kernel, extractPointer(args)...);

			if (cudaGetLastError() != cudaSuccess) {
				std::string message = "Failed to launch CUDA kernel: ";
				message += cudaGetErrorString(cudaGetLastError());
				throw std::runtime_error(message);
			}

		}

		template <typename... Args>	requires (IsBuffer<Args> && ...)
		void synchronize(Args&... readBack) {
			cudaError_t error = cudaDeviceSynchronize();
			if (error != cudaSuccess) {
				std::string message = "Failed to synchronize CUDA device: ";
				message += cudaGetErrorString(error);
				throw std::runtime_error(message);
			}

			([&](auto& buffer) {
				buffer.copyToHost();
			}(readBack), ...);

		}

};
