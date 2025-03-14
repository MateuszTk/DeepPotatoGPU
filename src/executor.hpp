#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "generic.hpp"

template <typename Func, typename... Args>
__global__ void run(Func func, Args... args) {
	func(args...);
}

class Executor {

	public:

		Executor() = default;
		virtual ~Executor() = default;

		template <typename Kernel, typename... Args>
		void executeGPU(unsigned int size, Args... args) {
			Kernel kernel{};

			run<<<1, size>>>(kernel, args...);
		}

		template <typename Kernel, typename... Args>
		void executeCPU(unsigned int size, Args... args) {
			Kernel kernel{};

			for (unsigned int i = 0; i < size; i++) {
				kernel(args...);
				kernel.threadIdxG.x++;
			}
		}

};
