#pragma once

#include "executor.hpp"

#if EXECUTOR_DEBUG_ON
#define EXECUTOR_CPU_LOG(format, ...) printf("[CPU EXECUTOR] " format, __VA_ARGS__)
#else
#define EXECUTOR_CPU_LOG(...)
#endif

class CPUExecutor : public Executor {

	public:

		CPUExecutor() = default;
		virtual ~CPUExecutor() = default;

		template <typename Kernel, typename... Args>
		void execute(dim3 threadsPerBlock, Args&... args) {
			EXECUTOR_CPU_LOG("Launching CPU kernel with arguments: %s\n", ARGS_TO_STRING(args));

			Kernel kernel{};

			for (unsigned int z = 0; z < threadsPerBlock.z; z++) {
				for (unsigned int y = 0; y < threadsPerBlock.y; y++) {
					for (unsigned int x = 0; x < threadsPerBlock.x; x++) {
						kernel.threadIdxG = { x, y, z };
						kernel(args...);
					}
				}
			}
		}

		void synchronize() override {
			// Nothing to do
		}

};
