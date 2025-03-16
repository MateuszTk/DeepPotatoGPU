#pragma once

#include "executor.hpp"

class CPUExecutor : public Executor {

	public:

		CPUExecutor() = default;
		virtual ~CPUExecutor() = default;

		template <typename Kernel, typename... Args>
		void execute(dim3 threadsPerBlock, Args&... args) {
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

		template <typename... Args>	requires ((IsBuffer<Args> || ContainsBuffer<Args>) && ...)
		void synchronize(Args&... readBack) {
			// Nothing to do
		}

};
