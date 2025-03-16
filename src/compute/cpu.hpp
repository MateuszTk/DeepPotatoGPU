#pragma once

#include "executor.hpp"

class CPUExecutor : public Executor {

	public:

		CPUExecutor() = default;
		virtual ~CPUExecutor() = default;

		template <typename Kernel, typename... Args>
		void execute(unsigned int size, Args&... args) {
			Kernel kernel{};

			for (unsigned int i = 0; i < size; i++) {
				kernel(args...);
				kernel.threadIdxG.x++;
			}
		}

		template <typename... Args>	requires ((IsBuffer<Args> || ContainsBuffer<Args>) && ...)
		void synchronize(Args&... readBack) {
			// Nothing to do
		}

};
