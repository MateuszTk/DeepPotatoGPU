#pragma once

#include "executor.hpp"
#include "timer.hpp"

#if EXECUTOR_DEBUG_ON
#define EXECUTOR_CPU_LOG(format, ...) printf("[CPU EXECUTOR] " format, __VA_ARGS__)
#else
#define EXECUTOR_CPU_LOG(...)
#endif

class CPUExecutor : public Executor {
	private:

		struct Worker {
			std::thread thread;
			std::condition_variable cv;
			std::condition_variable cvWait;
			std::queue<std::function<void()>> jobs;
			std::mutex mutex;
			bool idle = false;
			bool isRunning = true;

			Worker() {
				thread = std::thread(&Worker::run, this);
			}

			~Worker() {
				{
					std::unique_lock<std::mutex> lock(mutex);
					isRunning = false;
				}
				cv.notify_all();
				if (thread.joinable()) {
					thread.join();
				}
			}

			void run() {
				while (isRunning) {
					std::function<void()> job;
					{
						std::unique_lock<std::mutex> lock(mutex);
						cv.wait(lock, [this] { return !jobs.empty() || !isRunning; });
						if (!isRunning) break;
						job = std::move(jobs.front());
						jobs.pop();
					}
					job();
					{
						std::unique_lock<std::mutex> lock(mutex);
						if (jobs.empty()) {
							idle = true;
							cvWait.notify_all();
						}
					}
				}
			}

			template <typename Kernel, typename... Args>
			void addJob(dim3 start, dim3 stop, dim3 threadsPerBlock, Args&... args) {
				{
					std::unique_lock<std::mutex> lock(mutex);
					jobs.push([=, &args...]() {
						//Timer timer;
						Kernel kernel{};
						kernel.blockIdxG = { 0, 0, 0 };
						kernel.blockDimG = threadsPerBlock;
						for (unsigned int z = start.z; z < stop.z; z++) {
							for (unsigned int y = start.y; y < stop.y; y++) {
								for (unsigned int x = start.x; x < stop.x; x++) {
									kernel.threadIdxG = { x, y, z };
									kernel(args...);
								}
							}
						}
						//timer.stop();
						});
					idle = false;
				}
				cv.notify_one();
			}

			void wait() {
				std::unique_lock<std::mutex> lock(mutex);
				cvWait.wait(lock, [this] { return idle; });
			}

		};

		void wait() {
			for (auto& worker : workers) {
				worker.wait();
			}
		}

		std::vector<Worker> workers;

	public:

		CPUExecutor(uint32_t workerCount = 0) : workers(workerCount) {};
		virtual ~CPUExecutor() = default;

		template <typename Kernel, typename... Args>
		void execute(dim3 threadsPerBlock, Args&... args) {
			EXECUTOR_CPU_LOG("Launching CPU kernel with arguments: %s\n", ARGS_TO_STRING(args));
			EXECUTOR_CPU_LOG(" *  Threads per block: %u, %u, %u\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

			if (workers.size() > 1) {
				EXECUTOR_CPU_LOG(" *  Using %d workers\n", WORKER_COUNT);

				//Timer timerh;
				for (unsigned int i = 0; i < workers.size(); i++) {
					dim3 start = { 0, 0, 0 };
					dim3 stop = threadsPerBlock;
					if (threadsPerBlock.z > 1) {
						uint32_t zPerWorker = threadsPerBlock.z / workers.size();
						start.z = i * zPerWorker;
						stop.z = (i == workers.size() - 1) ? threadsPerBlock.z : (i + 1) * zPerWorker;
					}
					else {
						uint32_t yPerWorker = threadsPerBlock.y / workers.size();
						start.y = i * yPerWorker;
						stop.y = (i == workers.size() - 1) ? threadsPerBlock.y : (i + 1) * yPerWorker;
					}
					//timerh.start();
					workers[i].addJob<Kernel>(start, stop, threadsPerBlock, args...);
				}

				wait();

				//timerh.stop();

				//std::cout << '\n';
			}
			else {
				EXECUTOR_CPU_LOG(" *  Using main thread\n");
				Kernel kernel{};
				kernel.blockIdxG = { 0, 0, 0 };
				kernel.blockDimG = threadsPerBlock;

				for (unsigned int z = 0; z < threadsPerBlock.z; z++) {
					for (unsigned int y = 0; y < threadsPerBlock.y; y++) {
						for (unsigned int x = 0; x < threadsPerBlock.x; x++) {
							kernel.threadIdxG = { x, y, z };
							kernel(args...);
						}
					}
				}
			}
		}

		void synchronize() override {
			// Nothing to do
		}

};
