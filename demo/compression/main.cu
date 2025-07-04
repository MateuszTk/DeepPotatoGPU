
#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "timer.hpp"

#include "canvas.hpp"
#include "image.hpp"

void initInput(Matrix3D<float>& input, Canvas& canvas) {
	int index = 0;

	for (int y = 0; y < canvas.getHeight(); y++) {
		for (int x = 0; x < canvas.getWidth(); x++) {
			input(index, 0, 0) = (float)x / canvas.getWidth();
			input(index, 1, 0) = (float)y / canvas.getHeight();
			index++;
		}
	}
}

int main() {
	Image image("data/happybread.png");
	Canvas canvas(200, 200);
	Timer timer, timer2;

	CUDAExecutor exec;

	Network network({
			InputLayer(2),
			DenseLayer(30, Activation::Sigmoid),
			DenseLayer(20, Activation::Sigmoid),
			DenseLayer(10, Activation::Sigmoid),
			DenseLayer(3, Activation::Sigmoid)
		},
		canvas.getWidth() * canvas.getHeight(),
		32
	);

	srand(time(NULL));
	network.initialize();

	const int sets = 100;
	DataSet<float> trainingDataSet({ 0, 0 }, { 0, 0, 0 }, network.getMaximumTrainBatchSize() * sets);

	Matrix3D<float> testInput({ network.getMaximumBatchSize(), 2, 1 });
	testInput.getBuffer().setDirection(BufferDirection::HostToDevice);
	initInput(testInput, canvas);

	const int epochs = 1'000'000'000;

	for (int epoch = 0; epoch < epochs; epoch++) {
		for (int set = 0; set < sets; set++) {
			for (int i = 0; i < network.getMaximumTrainBatchSize(); i++) {
				int index = i + set * network.getMaximumTrainBatchSize();

				trainingDataSet.input(index, 0, 0) = (rand() / (float)RAND_MAX);
				trainingDataSet.input(index, 1, 0) = (rand() / (float)RAND_MAX);

				uint3 pixel = image.getPixel(trainingDataSet.input(index, 0, 0), trainingDataSet.input(index, 1, 0));

				trainingDataSet.output(index, 0, 0) = pixel.x / 255.0f;
				trainingDataSet.output(index, 1, 0) = pixel.y / 255.0f;
				trainingDataSet.output(index, 2, 0) = pixel.z / 255.0f;
			}
		}

		for (int set = 0; set < sets; set++) {
			network.forward(exec, trainingDataSet.input, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			network.backward(exec, trainingDataSet.output, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			network.update(exec, 0.1f, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
		}

		if (epoch % (100000 / (network.getMaximumTrainBatchSize() * sets)) == 0) {
			std::cout << "Epoch: " << epoch << ", Samples: " << sets * network.getMaximumTrainBatchSize() * epoch << "\n";
			std::cout << " * Training ";
			timer2.stop();
			timer2.start();

			network.forward(exec, testInput, network.getMaximumBatchSize());

			for (int i = 0; i < network.getMaximumBatchSize(); i++) {
				uint8_t colorR = (uint8_t)(network.getOutput()(i, 0, 0) * 255.0f);
				uint8_t colorG = (uint8_t)(network.getOutput()(i, 1, 0) * 255.0f);
				uint8_t colorB = (uint8_t)(network.getOutput()(i, 2, 0) * 255.0f);
				canvas.setPixel(i, colorR, colorG, colorB);
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
