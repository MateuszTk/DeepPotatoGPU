#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct GenericKernel {

	protected:

		uint3 threadIdxG;

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

	public:

		GenericKernel() = default;
		~GenericKernel() = default;

		friend class Executor;
		friend class CUDAExecutor;
		friend class CPUExecutor;

};

#define GENERIC_KERNEL(name) struct name : public GenericKernel
#define GENERIC_KERNEL_ENTRY __device__ __host__ void operator()