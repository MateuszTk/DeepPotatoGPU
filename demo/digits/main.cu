
#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "timer.hpp"

#include "canvas.hpp"
#include "idx.hpp"

class TestDigits {
private:
	IDX::IDX_Data testImages;
	IDX::IDX_Data testLabels;
	int testSampleId;
	std::unique_ptr<Matrix2D<float>> testInput;
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
		canvas.update();
		return canvas.frame();
	}
};

int main(int argc, char** argv) {
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

	//CPUExecutor exec;
	CUDAExecutor exec;

	Network network({
			InputLayer(imageSize),
			DenseLayer(128, Activation::Sigmoid),
			DenseLayer(128, Activation::Sigmoid),
			DenseLayer(10, Activation::Sigmoid)
		},
		30,
		30
	);

	srand(8888);
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

	const int epochs = 1'000'000'000;

	auto start = std::chrono::high_resolution_clock::now();
	int lastEpoch = -1;
	int testSampleId = 0;

	double forwardTotal = 0.0;
	double backwardTotal = 0.0;
	double updateTotal = 0.0;
	int iters = 0;

	for (int epoch = 0; epoch < epochs; epoch++) {
		for (int set = 0; set < sets; set++) {
			std::cout << "Forward: ";
			exec.synchronize();
			Timer timerf;
			network.forward(exec, trainingDataSet.input, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			exec.synchronize();
			forwardTotal += timerf.stop();

			std::cout << "Backward: ";
			Timer timerb;
			network.backward(exec, trainingDataSet.output, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			exec.synchronize();
			backwardTotal += timerb.stop();

			std::cout << "Update: ";
			Timer timeru;
			network.update(exec, 0.1f, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			exec.synchronize();
			updateTotal += timeru.stop();
			iters++;

			std::cout << "Forward time avg: " << forwardTotal / iters * 1000.0 << " ms, "
				<< "Backward time avg: " << backwardTotal / iters * 1000.0 << " ms, "
				<< "Update time avg: " << updateTotal / iters * 1000.0 << " ms\n";

			if (set % (sets / 10) == 0) {
				std::cout << "Epoch: " << epoch << ", Set: " << set << "/" << sets << "\n";
				auto elapsed = std::chrono::high_resolution_clock::now() - start;
				std::chrono::duration<double, std::milli> diff = elapsed;
				std::cout << " * Training speed: " << (set * network.getMaximumTrainBatchSize()) / diff.count() * 1000.0f << " samples/s\n";
				if (!tester.test(exec, network, canvas)) {
					epoch = epochs;
					break;
				}
			}
		}

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> diff = end - start;
		start = end;
		std::cout << "Epoch: " << epoch << ", Samples: " << sets * network.getMaximumTrainBatchSize() * (epoch + 1) << "\n";
		std::cout << " * Training speed: " << (epoch - lastEpoch) * sets * network.getMaximumTrainBatchSize() / diff.count() * 1000.0f << " samples/s\n";
		lastEpoch = epoch;
	}

	return 0;
}
