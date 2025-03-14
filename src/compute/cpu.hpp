#pragma once

#include "executor.hpp"

class CPUExecutor : public Executor {

	private:

		template <typename T>
		T* extractPointer(Buffer<T>& buffer) {
			return buffer.getDataHost();
		}

		template <typename T>
		T extractPointer(T other) {
			return other;
		}

	public:

		CPUExecutor() = default;
		virtual ~CPUExecutor() = default;

		template <typename Kernel, typename... Args>
		void execute(unsigned int size, Args&... args) {
			Kernel kernel{};

			for (unsigned int i = 0; i < size; i++) {
				kernel(extractPointer(args)...);
				kernel.threadIdxG.x++;
			}
		}

		template <typename... Args>	requires (IsBuffer<Args> && ...)
		void synchronize(Args&... readBack) {
			// Nothing to do
		}

};
