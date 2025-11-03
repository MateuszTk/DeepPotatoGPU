
#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "timer.hpp"

#include "canvas.hpp"
#include "image.hpp"

void initInput(Matrix2D<float>& input, Canvas& canvas) {
	int index = 0;

	for (int y = 0; y < canvas.getHeight(); y++) {
		for (int x = 0; x < canvas.getWidth(); x++) {
			input(index, 0) = (float)x / canvas.getWidth();
			input(index, 1) = (float)y / canvas.getHeight();
			index++;
		}
	}
}

int main() {
	Image image("data/happybread.png");
	Canvas canvas(200, 200);
	Timer timer, timer2;

	#ifdef CUDA_AVAILABLE
	CUDAExecutor exec;
	#else
	CPUExecutor exec;
	#endif

	Network network({
			InputLayer(2),
			DenseLayer(30, Activation::Sigmoid),
			DenseLayer(20, Activation::Sigmoid),
			DenseLayer(10, Activation::Sigmoid),
			DenseLayer(3, Activation::Sigmoid)
		},
		canvas.getWidth() * canvas.getHeight(),
		30
	);

	srand(time(NULL));
	network.initialize();

	const int sets = 1000;
	DataSet<float> trainingDataSet({ 0, 0 }, { 0, 0, 0 }, network.getMaximumTrainBatchSize() * sets);

	Matrix2D<float> testInput({ network.getMaximumBatchSize(), 2 });
	testInput.getBuffer().setDirection(BufferDirection::HostToDevice);
	initInput(testInput, canvas);

	const int iterations = 1'000'000'000;
	int lastIteration = 0;

	for (int iteration = 0; iteration < iterations; iteration++) {
		for (int set = 0; set < sets; set++) {
			for (int i = 0; i < network.getMaximumTrainBatchSize(); i++) {
				int index = i + set * network.getMaximumTrainBatchSize();

				trainingDataSet.input(index, 0) = (rand() / (float)RAND_MAX);
				trainingDataSet.input(index, 1) = (rand() / (float)RAND_MAX);

				uint3 pixel = image.getPixel(trainingDataSet.input(index, 0), trainingDataSet.input(index, 1));

				trainingDataSet.output(index, 0) = pixel.x / 255.0f;
				trainingDataSet.output(index, 1) = pixel.y / 255.0f;
				trainingDataSet.output(index, 2) = pixel.z / 255.0f;
			}
		}

		for (int set = 0; set < sets; set++) {
			network.forward(exec, trainingDataSet.input, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			network.backward(exec, trainingDataSet.output, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
			network.update(exec, 0.1f, network.getMaximumTrainBatchSize(), set * network.getMaximumTrainBatchSize());
		}

		if (iteration % (100000 / (network.getMaximumTrainBatchSize() * sets)) == 0) {
			auto elapsed = timer2.stop(false);
			std::cout << "Iteration: " << iteration << ", Samples: " << sets * network.getMaximumTrainBatchSize() * iteration << ", samples per second: "
				<< (sets * network.getMaximumTrainBatchSize() * (iteration - lastIteration)) / elapsed << "\n";
			std::cout << " * Training " << elapsed << " ms\n";
			lastIteration = iteration;
			
			timer2.start();

			network.forward(exec, testInput, network.getMaximumBatchSize());

			for (int i = 0; i < network.getMaximumBatchSize(); i++) {
				uint8_t colorR = (uint8_t)(static_cast<float>(network.getOutput()(i, 0)) * 255.0f);
				uint8_t colorG = (uint8_t)(static_cast<float>(network.getOutput()(i, 1)) * 255.0f);
				uint8_t colorB = (uint8_t)(static_cast<float>(network.getOutput()(i, 2)) * 255.0f);
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
