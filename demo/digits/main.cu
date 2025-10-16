
#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "timer.hpp"

#include "canvas.hpp"
#include "idx.hpp"
#include "../utils/stats.hpp"

class TestDigits {
private:
	IDX::IDX_Data testImages;
	IDX::IDX_Data testLabels;
	int testSampleId;
	std::unique_ptr<Matrix2D<float>> testInput;
	std::unique_ptr<Matrix2D<float>> fullInput;
	unsigned int imageSize;
	int width;
	int height;

public:
	TestDigits(const std::string& imagesFile, const std::string& labelsFile) :
		testImages(IDX::import(imagesFile)),
		testLabels(IDX::import(labelsFile)),
		testSampleId(0) {
		width = testImages.header.sizes[1];
		height = testImages.header.sizes[2];
		imageSize = width * height;

		const std::array<unsigned int, 2>& dimensions = { 1, imageSize };
		testInput = std::make_unique<Matrix2D<float>>(dimensions);
		testInput->getBuffer().setDirection(BufferDirection::HostToDevice);

		const std::array<unsigned int, 2>& fullDimensions = { testImages.header.sizes[0], imageSize };
		fullInput = std::make_unique<Matrix2D<float>>(fullDimensions);
		fullInput->getBuffer().setDirection(BufferDirection::HostToDevice);
		for (int i = 0; i < testImages.header.sizes[0]; i++) {
			const uint8_t* image = testImages.data + i * imageSize;
			for (int j = 0; j < imageSize; j++) {
				(*fullInput)(i, j) = image[j] / 255.0f;
			}
		}
	}

	template <typename Executor>
	bool test(Executor& exec, Network& network, Canvas& canvas) {
		const uint8_t* testImage = testImages.data + testSampleId * imageSize;
		for (int j = 0; j < imageSize; j++) {
			(*testInput)(0, j) = testImage[j] / 255.0f;
		}
		int testLabel = testLabels.data[testSampleId];
		network.forward(exec, *testInput, 1);

		int maxIndex = 0;
		float maxValue = network.getOutput()(0, 0);
		for (int j = 1; j < 10; j++) {
			if (static_cast<float>(network.getOutput()(0, j)) > maxValue) {
				maxValue = network.getOutput()(0, j);
				maxIndex = j;
			}
		}
		for (int j = 0; j < 10; j++) {
			std::cout << j << " " << std::fixed << std::setprecision(2) << static_cast<float>(network.getOutput()(0, j))
				<< (j == testLabel ? " *" : "  ") << (j == maxIndex ? " <" : "  ") << "\n";
		}

		for (int y = 0; y < canvas.getHeight(); y++) {
			for (int x = 0; x < canvas.getWidth(); x++) {
				int imageX = x * width / canvas.getWidth();
				int imageY = y * height / canvas.getHeight();
				uint8_t color = (uint8_t)((*testInput)(0, imageX + imageY * width) * 255.0f);
				canvas.setPixel(x / (float)canvas.getWidth(), y / (float)canvas.getHeight(), color, color, color);
			}
		}
		testSampleId++;
		if (testSampleId >= testImages.header.sizes[0]) {
			testSampleId = 0;
		}
		canvas.update();
		return canvas.frame();
	}

	template <typename Executor>
	float testAll(Executor& exec, Network& network) {		
		int correct = 0;
		for (int input = 0; input < testImages.header.sizes[0]; input += network.getMaximumBatchSize()) {
			int batchSize = std::min(network.getMaximumBatchSize(), testImages.header.sizes[0] - input);
			network.forward(exec, *fullInput, batchSize, input);
			for (int i = 0; i < batchSize; i++) {
				int testLabel = testLabels.data[input + i];
				int maxIndex = 0;
				float maxValue = network.getOutput()(i, 0);
				for (int j = 1; j < 10; j++) {
					if (static_cast<float>(network.getOutput()(i, j)) > maxValue) {
						maxValue = network.getOutput()(i, j);
						maxIndex = j;
					}
				}
				if (maxIndex == testLabel) {
					correct++;
				}
				//std::cout << "Sample " << (input + i) << ": Label: " << testLabel << ", Predicted: " << maxIndex << '\n';
			}
		}

		float accuracy = static_cast<float>(correct) / testImages.header.sizes[0];
		std::cout << "Test accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "%\n";
		return accuracy;
	}
};

struct TestConfig {
	std::string logDir = "scale_test/";
	std::vector<uint32_t> workerCounts = { 4 };// 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	std::vector<uint32_t> hiddenLayerSizes = { 128 }; //{ 4, 6, 8, 10, 12, 14, 16, 18, 20, 28, 32, 64, 128, 256, 512, 768, 1024 };
	int epochs = 10;
	int batchSize = 30;
	int iterations = 1;
	bool testSet = false;
	bool verbose = false;

	bool configure(const std::vector<std::string>& args) {
		auto parseList = [](const std::string& str) {
			std::vector<uint32_t> values;
			size_t pos = 0;
			std::string s = str;
			while ((pos = s.find(',')) != std::string::npos) {
				values.push_back(std::stoi(s.substr(0, pos)));
				s.erase(0, pos + 1);
			}
			if (!s.empty()) {
				values.push_back(std::stoi(s));
			}
			return values;
			};

		for (size_t i = 1; i < args.size(); i++) {
			if (args[i] == "--log-dir" && i + 1 < args.size()) {
				logDir = args[i + 1];
				if (logDir.back() != '/' && logDir.back() != '\\') {
					logDir += '/';
				}
				i++;
			}
			else if (args[i] == "--workers" && i + 1 < args.size()) {
				workerCounts.clear();
				std::string workersStr = args[i + 1];
				workerCounts = parseList(workersStr);
				i++;
			}
			else if (args[i] == "--hidden-layers" && i + 1 < args.size()) {
				hiddenLayerSizes.clear();
				std::string layersStr = args[i + 1];
				hiddenLayerSizes = parseList(layersStr);
				i++;
			}
			else if (args[i] == "--epochs" && i + 1 < args.size()) {
				epochs = std::stoi(args[i + 1]);
				i++;
			}
			else if (args[i] == "--batch-size" && i + 1 < args.size()) {
				batchSize = std::stoi(args[i + 1]);
				i++;
			}
			else if (args[i] == "--iterations" && i + 1 < args.size()) {
				iterations = std::stoi(args[i + 1]);
				i++;
			}
			else if (args[i] == "--test-set") {
				testSet = true;
			}
			else if (args[i] == "--verbose") {
				verbose = true;
			}
			else if (args[i] == "--help" || args[i] == "-h") {
				std::cout << "Usage: digits [options]\n";
				std::cout << "Options:\n";
				std::cout << "  --log-dir <dir>          Directory to save log files\n";
				std::cout << "  --workers <list>         Comma-separated list of CPU worker counts to test\n";
				std::cout << "  --hidden-layers <list>   Comma-separated list of hidden layer sizes to test\n";
				std::cout << "  --epochs <num>           Number of epochs to train\n";
				std::cout << "  --batch-size <num>       Training batch size\n";
				std::cout << "  --iterations <num>       Number of test iterations to run\n";
				std::cout << "  --test-set               Evaluate on test set during training\n";
				std::cout << "  --verbose                Enable verbose output\n";
				std::cout << "  --help, -h               Show this help message\n";
				return false;
			}
			else {
				std::cout << "Bad argument: " << args[i] << '\n';
				return false;
			}
		}
		return true;
	}

	void print(std::ostream& os) const {
		os << "Test configuration:\n";
		os << " * Log directory: " << logDir << '\n';
		os << " * Worker counts: ";
		for (size_t i = 0; i < workerCounts.size(); i++) {
			os << workerCounts[i];
			if (i + 1 < workerCounts.size()) {
				os << ", ";
			}
		}
		os << '\n';
		os << " * Hidden layer sizes: ";
		for (size_t i = 0; i < hiddenLayerSizes.size(); i++) {
			os << hiddenLayerSizes[i];
			if (i + 1 < hiddenLayerSizes.size()) {
				os << ", ";
			}
		}
		os << '\n';
		os << " * Epochs: " << epochs << '\n';
		os << " * Batch size: " << batchSize << '\n';
		os << " * Test set: " << (testSet ? "true" : "false") << '\n';
		os << " * Verbose: " << (verbose ? "true" : "false") << '\n';
	}
};
 
int main(int argc, char** argv) {
	TestConfig config;
	bool canContinue = config.configure(std::vector<std::string>(argv, argv + argc));
	if (!canContinue) {
		return 0;
	}
	config.print(std::cout);
	std::cout << '\n';
	std::cout << "lowp_t: " << typeid(lowp_t).name() << '\n';
	std::cout << "USE_WMMA: " << (USE_WMMA ? "true" : "false") << '\n';

	#ifdef CUDA_AVAILABLE
	std::unique_ptr<Stats> stats = std::make_unique<CUDAStats>();
	#else
	std::unique_ptr<Stats> stats = std::make_unique<CPUStats>();
	#endif

	for (auto cpuWorkerCount : config.workerCounts) {
		for (auto hiddenLayerSize : config.hiddenLayerSizes) {
			std::cout << "Testing with " << cpuWorkerCount << " CPU workers and hidden layer size " << hiddenLayerSize << '\n';

			#ifdef CUDA_AVAILABLE
			CUDAExecutor exec;
			#else
			CPUExecutor exec(cpuWorkerCount);
			#endif

			for (int testIter = 0; testIter < config.iterations; testIter++) {
				std::cout << "Test iteration: " << testIter << '\n';

				IDX::IDX_Data trainImages = IDX::import("data/train-images.idx3-ubyte");
				IDX::printData(trainImages);
				IDX::IDX_Data trainLabels = IDX::import("data/train-labels.idx1-ubyte");
				IDX::printData(trainLabels);

				const int numImages = trainImages.header.sizes[0];
				const int width = trainImages.header.sizes[1];
				const int height = trainImages.header.sizes[2];
				const int imageSize = width * height;
				TestDigits tester("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

				Canvas canvas(200, 200);

				stats->resetEnergyConsumption();

				Network network({
						InputLayer(imageSize),
						DenseLayer(hiddenLayerSize, Activation::Sigmoid),
						DenseLayer(hiddenLayerSize, Activation::Sigmoid),
						DenseLayer(10, Activation::Sigmoid)
					},
					config.batchSize,
					config.batchSize
				);

				std::string logName = config.logDir + "log_digits_";
				logName += std::to_string(testIter) + "_";
				for (int i = 0; i < network.getLayerCount(); i++) {
					logName += std::to_string(network.getLayerType(i).getNeurons()) + "_";
				}
				logName += std::to_string(network.getMaximumTrainBatchSize()) + "_";
				logName += std::to_string(cpuWorkerCount) + "_";
				logName += std::to_string(config.testSet) + "_";
				logName += std::string(typeid(lowp_t).name()) + "_";
				logName += std::to_string(USE_WMMA) + "_";
				logName += std::string(typeid(exec).name()) + "_";
				logName += ".txt";
				std::cout << "Log file: " << logName << '\n';
				std::ofstream logFile(logName);
				if (!logFile.is_open()) {
					std::cerr << "Error: could not open log file\n";
					return 1;
				}

				//srand(8888);
				srand(time(nullptr));
				network.initialize();

				const int sets = numImages / network.getMaximumTrainBatchSize();

				DataSet<float> trainingDataSet(imageSize, 10, numImages);
				trainingDataSet.output.fill(0.0f);
				for (int i = 0; i < numImages; i++) {
					const uint8_t* image = trainImages.data + i * imageSize;
					for (int j = 0; j < imageSize; j++) {
						trainingDataSet.input(i, j) = image[j] / 255.0f;
					}
					trainingDataSet.output(i, trainLabels.data[i]) = 1.0f;
				}

				int lastEpoch = -1;
				int testSampleId = 0;

				double forwardTotal = 0.0;
				double backwardTotal = 0.0;
				double updateTotal = 0.0;
				int iters = 0;
				Timer setTimer;
				Timer epochTimer;
				Timer totalTimer;

				Matrix1D<float> lossMat({ network.getMaximumBatchSize() });
				lossMat.getBuffer().setDirection(BufferDirection::DeviceToHost);
				float loss = 0.0f;

				for (int epoch = 0; epoch < config.epochs; epoch++) {
					for (int set = 0; set < sets; set++) {
						exec.synchronize();
						Timer timerf;
						network.forward(exec, trainingDataSet.input, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
						exec.synchronize();
						forwardTotal += timerf.stop(false);

						Timer timerb;
						network.backward(exec, trainingDataSet.output, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
						exec.synchronize();
						backwardTotal += timerb.stop(false);

						Timer timeru;
						network.update(exec, 0.1f, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
						exec.synchronize();
						updateTotal += timeru.stop(false);
						iters++;

						if (config.testSet) {
							//loss += network.loss(exec, lossMat, trainingDataSet.output, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
						}

						if (set % (sets / 10) == 0) {
							float diffMs = setTimer.stop(false) * 1000.0f;

							int samplesTotal = (set + 1) * network.getMaximumTrainBatchSize() + epoch * sets * network.getMaximumTrainBatchSize();

							std::cout << "Epoch: " << epoch << ", Set: " << set << "/" << sets << ", Samples: "
								<< (set + 1) * network.getMaximumTrainBatchSize() << "/" << sets * network.getMaximumTrainBatchSize() << "\n";
							std::cout << "Power usage: " << stats->getPowerUsage() << " W\n";
							std::cout << "Energy usage: " << stats->getEnergyConsumption() << " Wh\n";

							logFile << samplesTotal << " " << totalTimer.stop(false) * 1000.0f << " " << forwardTotal * 1000.0f << " " << backwardTotal * 1000.0f << " " << updateTotal * 1000.0f;
							if (config.testSet) {
								logFile << " " << tester.testAll(exec, network);
								logFile << " " << std::fixed << std::setprecision(8) << loss / (sets / 10);
								std::cout << "Loss: " << std::fixed << std::setprecision(8) << loss / (sets / 10) << "\n";
								loss = 0.0f;
							}
							else {
								logFile << " 0.0 0.0";
							}
							logFile << "\n";

							if (config.verbose) {
								//std::cout << " * Training speed: " << (set * network.getMaximumTrainBatchSize()) / diff_ms << " samples/s\n";

								std::cout << "Forward time avg: " << forwardTotal / iters * 1000.0 << " ms, "
									<< "Backward time avg: " << backwardTotal / iters * 1000.0 << " ms, "
									<< "Update time avg: " << updateTotal / iters * 1000.0 << " ms\n";

								if (!tester.test(exec, network, canvas)) {
									epoch = config.epochs;
									break;
								}
							}

							setTimer.start();
						}
					}

					std::cout << "Epoch: " << epoch << ", Samples: " << sets * network.getMaximumTrainBatchSize() * (epoch + 1) << "\n";
					std::cout << " * Training speed: " << (epoch - lastEpoch) * sets * network.getMaximumTrainBatchSize() / epochTimer.stop(false) << " samples/s\n";
					lastEpoch = epoch;
					stats->tick();
					epochTimer.start();
				}

				float energyFinal = stats->getEnergyConsumption();
				float timeFinal = totalTimer.stop(false);
				std::cout << "Total energy: " << energyFinal << " Wh\n";
				std::cout << "Total time: " << timeFinal << " s\n";
				std::cout << "Average power: " << (energyFinal * 3600.0f) / timeFinal << " W\n";

				tester.testAll(exec, network);

				logFile.close();
			}
		}
	}
	return 0;
}
