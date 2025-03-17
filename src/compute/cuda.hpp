#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "executor.hpp"
#include "buffer.hpp"

#if EXECUTOR_DEBUG_ON
#define EXECUTOR_CUDA_LOG(...) printf("[CUDA EXECUTOR] "__VA_ARGS__)
#else
#define EXECUTOR_CUDA_LOG(...)
#endif

template <typename Func, typename... Args>
__global__ void run(Func func, Args... args) {
	func(args...);
}

class CUDAExecutor : public Executor {

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
		void execute(dim3 threadsPerBlock, Args&... args) {
			Kernel kernel{};

			([&](auto& buffer) {
				if constexpr (IsBuffer<decltype(buffer)>) {
					buffer.transitionLocation(Location::Device);
				}

				if constexpr (ContainsBuffer<decltype(buffer)>) {
					buffer.getBuffer().transitionLocation(Location::Device);
				}
			}(args), ...);

			EXECUTOR_CUDA_LOG("Launching CUDA kernel with arguments: %s\n", ARGS_TO_STRING(args));

			run<<<1, threadsPerBlock>>>(kernel, args...);

			if (cudaGetLastError() != cudaSuccess) {
				std::string message = "Failed to launch CUDA kernel: ";
				message += cudaGetErrorString(cudaGetLastError());
				throw std::runtime_error(message);
			}

		}

		void synchronize() override {
			cudaError_t error = cudaDeviceSynchronize();
			if (error != cudaSuccess) {
				std::string message = "Failed to synchronize CUDA device: ";
				message += cudaGetErrorString(error);
				throw std::runtime_error(message);
			}
		}

};
