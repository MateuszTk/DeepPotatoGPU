
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
	Timer timer, timer2;

	CPUExecutor exec;

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

	srand(time(NULL));
	network.initialize();

	DataSet<float> trainingDataSet(imageSize, 10, numImages);
	trainingDataSet.output.fill(0.0f);
	for (int i = 0; i < numImages; i++) {
		for (int j = 0; j < imageSize; j++) {
			const uint8_t* image = trainImages.data + i * imageSize;
			trainingDataSet.input(i, j, 0) = image[j] / 255.0f;
		}
		trainingDataSet.output(i, trainLabels.data[i], 0) = 1.0f;
	}

	Matrix3D<float> testInput({ 1, (unsigned int)imageSize, 1 });
	testInput.getBuffer().setDirection(BufferDirection::HostToDevice);

	const int epochs = 1'000'000'000;

	for (int epoch = 0; epoch < epochs; epoch++) {
		for (int set = 0; set < sets; set++) {
			network.forward(exec, trainingDataSet.input, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			network.backward(exec, trainingDataSet.output, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			network.update(exec, 0.1f, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
		}

		if (epoch % (100 / (network.getMaximumTrainBatchSize() * sets)) == 0) {
			std::cout << "Epoch: " << epoch << ", Samples: " << sets * network.getMaximumTrainBatchSize() * epoch << "\n";
			std::cout << " * Training ";
			timer2.stop();
			timer2.start();

			for (int i = 0; i < network.getMaximumBatchSize(); i++) {
				for (int j = 0; j < imageSize; j++) {
					testInput(i, j, 0) = trainingDataSet.input(i + network.getMaximumTrainBatchSize(), j, 0);
				}
			}
			network.forward(exec, testInput, network.getMaximumBatchSize());

			for (int i = 0; i < network.getMaximumBatchSize(); i++) {
				for (int j = 0; j < 10; j++) {
					std::cout << network.getOutput()(i, j, 0) << "\n";
				}

				for (int y = 0; y < canvas.getHeight(); y++) {
					for (int x = 0; x < canvas.getWidth(); x++) {
						int imageX = x * width / canvas.getWidth();
						int imageY = y * height / canvas.getHeight();
						uint8_t color = (uint8_t)(testInput(i, imageX + imageY * width, 0) * 255.0f);
						canvas.setPixel(x / (float)canvas.getWidth(), y / (float)canvas.getHeight(), color, color, color);
					}
				}
			}

			canvas.update();
			if (!canvas.frame()) {
				break;
			}

			std::cout << " * Test ";
			timer2.stop();
			timer2.start();
		}
	}

	timer.stop();

	return 0;
}
