#pragma once

#include "external.hpp"

class Timer {

	private:

		std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

	public:

		Timer() {
			start();
		}

		void start() {
			startTime = std::chrono::high_resolution_clock::now();
		}

		void stop() {
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - startTime);
			std::cout << "Time: " << duration.count() / 1000.0f << "ms" << std::endl;
		}
};
