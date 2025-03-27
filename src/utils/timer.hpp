#pragma once

#include "external.hpp"

class Timer {

	private:

		std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

	public:

		Timer() {
			startTime = std::chrono::high_resolution_clock::now();
		}

		void start() {
			startTime = std::chrono::high_resolution_clock::now();
		}

		void stop() {
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - startTime);
			std::cout << "Time: " << duration.count() << "ms" << std::endl;
		}
};
