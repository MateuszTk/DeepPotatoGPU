
#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "timer.hpp"

#include "canvas.hpp"
#include "idx.hpp"

int main(int argc, char** argv) {
	IDX::IDX_Data trainImages = IDX::import("data/train-images.idx3-ubyte");
	IDX::printData(trainImages);
	IDX::IDX_Data trainLabels = IDX::import("data/train-labels.idx1-ubyte");
	IDX::printData(trainLabels);

	const int numImages = trainImages.header.sizes[0];
	const int width = trainImages.header.sizes[1];
	const int height = trainImages.header.sizes[2];
	const int imageSize = width * height;

	Canvas canvas(200, 200);

	CPUExecutor exec;
	//CUDAExecutor exec;

	const int sets = 1;

	Network network({
			InputLayer(imageSize),
			DenseLayer(100, Activation::Sigmoid),
			DenseLayer(100, Activation::Sigmoid),
			DenseLayer(10, Activation::Sigmoid)
		},
		1,
		1
	);

	srand(8888);
	network.initialize();

	DataSet<float> trainingDataSet(imageSize, 10, 1);//numImages);
	/*trainingDataSet.output.fill(0.0f);
	for (int i = 0; i < numImages; i++) {
		for (int j = 0; j < imageSize; j++) {
			const uint8_t* image = trainImages.data + i * imageSize;
			trainingDataSet.input(i, j, 0) = image[j] / 255.0f;
		}
		trainingDataSet.output(i, trainLabels.data[i], 0) = 1.0f;
	}*/

	Matrix3D<float> testInput({ 1, (unsigned int)imageSize, 1 });
	testInput.getBuffer().setDirection(BufferDirection::HostToDevice);

	const int epochs = 1'000'000'000;

	auto start = std::chrono::high_resolution_clock::now();
	int lastEpoch = 0;
	int testSampleId = 0;

	for (int epoch = 0; epoch < epochs; epoch++) {
		int i = epoch % numImages;
		const uint8_t* image = trainImages.data + i * imageSize;
		for (int j = 0; j < imageSize; j++) {
			trainingDataSet.input(0, j, 0) = image[j] / 255.0f;
		}
		trainingDataSet.output.fill(0.0f);
		trainingDataSet.output(0, trainLabels.data[i], 0) = 1.0f;

		for (int set = 0; set < sets; set++) {
			network.forward(exec, trainingDataSet.input, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			network.backward(exec, trainingDataSet.output, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			network.update(exec, 0.1f, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
		}

		if (epoch % 100 == 0) {
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> diff = end - start;

			if (diff.count() >= 1000) {
				start = end;

				std::cout << "Epoch: " << epoch << ", Samples: " << sets * network.getMaximumTrainBatchSize() * epoch << "\n";
				std::cout << " * Training speed: " << (epoch - lastEpoch) * sets * network.getMaximumTrainBatchSize() / (diff.count() / 1000.0f) << " samples/s\n";
				lastEpoch = epoch;

				testSampleId++;
				const uint8_t* testImage = trainImages.data + testSampleId * imageSize;
				for (int j = 0; j < imageSize; j++) {
					testInput(0, j, 0) = testImage[j] / 255.0f;
				}
				int testLabel = trainLabels.data[testSampleId];
				network.forward(exec, testInput, 1);

				int maxIndex = 0;
				float maxValue = network.getOutput()(0, 0, 0);
				for (int j = 1; j < 10; j++) {
					if (network.getOutput()(0, j, 0) > maxValue) {
						maxValue = network.getOutput()(0, j, 0);
						maxIndex = j;
					}
				}
				for (int j = 0; j < 10; j++) {
					std::cout << j << " " << std::fixed << std::setprecision(2) << network.getOutput()(0, j, 0) << (j == testLabel ? " *" : "  ") << (j == maxIndex ? " <" : "  ") << "\n";
				}

				for (int y = 0; y < canvas.getHeight(); y++) {
					for (int x = 0; x < canvas.getWidth(); x++) {
						int imageX = x * width / canvas.getWidth();
						int imageY = y * height / canvas.getHeight();
						uint8_t color = (uint8_t)(testInput(0, imageX + imageY * width, 0) * 255.0f);
						canvas.setPixel(x / (float)canvas.getWidth(), y / (float)canvas.getHeight(), color, color, color);
					}
				}

				canvas.update();
				if (!canvas.frame()) {
					break;
				}

				std::cout << " * Test ";
			}
		}
	}

	return 0;
}
