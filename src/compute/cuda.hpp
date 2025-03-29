#pragma once

#include "external.hpp"

#include "executor.hpp"
#include "buffer.hpp"

#ifdef CUDA_AVAILIABLE

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

	private:

		cudaDeviceProp prop;

	public:

		CUDAExecutor() {
			if (cudaSetDevice(0) != cudaSuccess) {
				throw std::runtime_error("Failed to set CUDA device");
			}

			
			if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
				throw std::runtime_error("Failed to get CUDA device properties");
			}

			EXECUTOR_CUDA_LOG("CUDA device: %s\n", prop.name);
		}

		virtual ~CUDAExecutor() {
			if (cudaDeviceReset() != cudaSuccess) {
				throw std::runtime_error("Failed to reset CUDA device");
			}
		}

		template <typename Kernel, typename... Args>
		void execute(dim3 threads, Args&... args) {
			Kernel kernel{};

			([&](auto& buffer) {
				if constexpr (IsBuffer<decltype(buffer)>) {
					buffer.transitionLocation(Location::Device);
				}

				if constexpr (ContainsBuffer<decltype(buffer)>) {
					buffer.getBuffer().transitionLocation(Location::Device);
				}
			}(args), ...);

			unsigned int maxThreads = prop.maxThreadsPerBlock;
			dim3 maxBlockDim = { (unsigned int)prop.maxThreadsDim[0], (unsigned int)prop.maxThreadsDim[1], (unsigned int)prop.maxThreadsDim[2] };
			dim3 threadsPerBlock;
			threadsPerBlock.x = std::min(threads.x, std::min(maxThreads, maxBlockDim.x));
			threadsPerBlock.y = std::max(std::min(threads.y, std::min(maxThreads / threadsPerBlock.x, maxBlockDim.y)), 1u);
			threadsPerBlock.z = std::max(std::min(threads.z, std::min(maxThreads / (threadsPerBlock.x * threadsPerBlock.y), maxBlockDim.z)), 1u);

			dim3 numBlocks;
			numBlocks.x = (threads.x + threadsPerBlock.x - 1) / threadsPerBlock.x;
			numBlocks.y = (threads.y + threadsPerBlock.y - 1) / threadsPerBlock.y;
			numBlocks.z = (threads.z + threadsPerBlock.z - 1) / threadsPerBlock.z;

			EXECUTOR_CUDA_LOG("Launching CUDA kernel with arguments: %s\n", ARGS_TO_STRING(args));
			EXECUTOR_CUDA_LOG(" *  Requested threads: %d, %d, %d\n", threads.x, threads.y, threads.z);
			EXECUTOR_CUDA_LOG(" *  Launching threads: %d, %d, %d\n", threadsPerBlock.x * numBlocks.x, threadsPerBlock.y * numBlocks.y, threadsPerBlock.z * numBlocks.z);
			EXECUTOR_CUDA_LOG(" *  Threads per block: %d, %d, %d\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
			EXECUTOR_CUDA_LOG(" *  Blocks:            %d, %d, %d\n", numBlocks.x, numBlocks.y, numBlocks.z);

			run<<<numBlocks, threadsPerBlock>>>(kernel, args...);

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

#endif
