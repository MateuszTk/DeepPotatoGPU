#pragma once

#include "external.hpp"

struct GenericKernel {

	protected:

		uint3 threadIdxG;
		uint3 blockIdxG;
		uint3 blockDimG;

	protected:

		__device__ __host__ float gfma(float a, float b, float c) {
			#ifdef __CUDA_ARCH__
				return fma(a, b, c);
			#else
				return a * b + c;
			#endif
		}

		__device__ __host__ uint3 getThreadIdx() {
			#ifdef __CUDA_ARCH__
				return threadIdx;
			#else
				return threadIdxG;
			#endif
		}

		__device__ __host__ uint3 getBlockIdx() {
			#ifdef __CUDA_ARCH__
			return blockIdx;
			#else
			return blockIdxG;
			#endif
		}

		__device__ __host__ uint3 getBlockDim() {
			#ifdef __CUDA_ARCH__
				return blockDim;
			#else
				return blockDimG;
			#endif
		}

	public:

		GenericKernel() = default;
		~GenericKernel() = default;

		friend class Executor;
		friend class CUDAExecutor;
		friend class CPUExecutor;

};

#define GENERIC_KERNEL(name) struct name : public GenericKernel
#define GENERIC_KERNEL_ENTRY __device__ __host__ void operator()