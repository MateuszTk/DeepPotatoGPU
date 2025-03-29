
#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "utils/timer.hpp"

#include "window.hpp"
#include "utils/image.hpp"

void xorDemo();
void image();

int main() {
	xorDemo();

	return 0;
}

std::array<DataSet<float>, 4> data = {{
	{{ 0, 0 }, { 0 }},
	{{ 0, 1 }, { 1 }},
	{{ 1, 0 }, { 1 }},
	{{ 1, 1 }, { 0 }}
}};

void xorDemo() {

	Window window(400, 400);

	CUDAExecutor exec;

	Timer timer, timer2;

	Network network({
		InputLayer(2),
		DenseLayer(3, Activation::Sigmoid),
		DenseLayer(1, Activation::Sigmoid)
		}, 400 * 400);

	srand(time(NULL));
	network.initialize();

	Matrix3D<float> input({ network.getMaximumBatchSize(), 2, 1 });

	for (int i = 0; i < 4000; i++) {
		for (DataSet<float>& dataSet : data) {
			network.forward(exec, dataSet.input);
			network.backward(exec, dataSet.output);
			network.update(exec, 0.5f);
		}

		if (i % 100 == 0) {
			std::cout << "Epoch: " << i << "\n";
			std::cout << "Training ";
			timer2.stop();
			timer2.start();

			int index = 0;

			for (int y = 0; y < window.getHeight(); y++) {
				for (int x = 0; x < window.getWidth(); x++) {

					input(index, 0, 0) = (float)x / window.getWidth();
					input(index, 1, 0) = (float)y / window.getHeight();

					index++;

					if (index == network.getMaximumBatchSize()) {

						network.forward(exec, input);

						int startIdx = x + y * window.getWidth() - network.getMaximumBatchSize() + 1;

						for (int i = 0; i < network.getMaximumBatchSize(); i++) {
							uint8_t color = (uint8_t)(network.getOutput()(i, 0, 0) * 255.0f);
							window.setPixel(startIdx + i, color, color, color);
						}

						index = 0;
					}
				}
			}

			window.update();
			if (!window.frame()) {
				break;
			}

			std::cout << "Test ";
			timer2.stop();
			timer2.start();
		}
	}


	for (DataSet<float>& dataSet : data) {
		network.forward(exec, dataSet.input);
		std::cout << "Input: " << dataSet.input << " Output: " << network.getOutput()(0, 0, 0) << "\n";
	}

	timer.stop();
}

void image() {

	Image image("data/happybread.png");

	Window window(200, 200);

	CPUExecutor exec;

	Timer timer, timer2;

	Network network({
		InputLayer(2),
		DenseLayer(20, Activation::Sigmoid),
		DenseLayer(10, Activation::Sigmoid),
		DenseLayer(3, Activation::Sigmoid)
		}, window.getWidth() * window.getHeight());

	srand(time(NULL));
	network.initialize();

	Matrix3D<float> testInput({ network.getMaximumBatchSize(), 2, 1 });
	DataSet<float> trainingDataSet({ 0, 0 }, { 0, 0, 0 });

	for (int i = 0; i < 1'000'000'000; i++) {
		trainingDataSet.input(0, 0, 0) = (rand() / (float)RAND_MAX);
		trainingDataSet.input(0, 1, 0) = (rand() / (float)RAND_MAX);
		uint3 pixel = image.getPixel(trainingDataSet.input(0, 0, 0), trainingDataSet.input(0, 1, 0));
		trainingDataSet.output(0, 0, 0) = pixel.x / 255.0f;
		trainingDataSet.output(0, 1, 0) = pixel.y / 255.0f;
		trainingDataSet.output(0, 2, 0) = pixel.z / 255.0f;

		network.forward(exec, trainingDataSet.input);
		network.backward(exec, trainingDataSet.output);
		network.update(exec, 0.3f);

		if (i % 100000 == 0) {
			std::cout << "Epoch: " << i << "\n";
			std::cout << "Training ";
			timer2.stop();
			timer2.start();

			int index = 0;

			for (int y = 0; y < window.getHeight(); y++) {
				for (int x = 0; x < window.getWidth(); x++) {

					testInput(index, 0, 0) = (float)x / window.getWidth();
					testInput(index, 1, 0) = (float)y / window.getHeight();

					index++;

					if (index == network.getMaximumBatchSize()) {

						network.forward(exec, testInput);

						int startIdx = x + y * window.getWidth() - network.getMaximumBatchSize() + 1;

						for (int i = 0; i < network.getMaximumBatchSize(); i++) {
							uint8_t colorR = (uint8_t)(network.getOutput()(i, 0, 0) * 255.0f);
							uint8_t colorG = (uint8_t)(network.getOutput()(i, 1, 0) * 255.0f);
							uint8_t colorB = (uint8_t)(network.getOutput()(i, 2, 0) * 255.0f);
							window.setPixel(startIdx + i, colorR, colorG, colorB);
						}

						index = 0;
					}
				}
			}

			window.update();
			if (!window.frame()) {
				break;
			}

			std::cout << "Test ";
			timer2.stop();
			timer2.start();
		}
	}

	timer.stop();
}