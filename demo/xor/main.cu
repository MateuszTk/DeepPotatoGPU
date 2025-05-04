#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "timer.hpp"

#include "canvas.hpp"

void xorDemo();

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

	Canvas canvas(400, 400);

	CUDAExecutor exec;

	Timer timer, timer2;

	Network network({
			InputLayer(2),
			DenseLayer(3, Activation::Sigmoid),
			DenseLayer(1, Activation::Sigmoid)
		}, 
		400 * 400,
		data.size()
	);

	srand(time(NULL));
	network.initialize();

	Matrix3D<float> input({ network.getMaximumBatchSize(), 2, 1 });

	Matrix3D<float> trainInput({ network.getMaximumTrainBatchSize(), 2, 1 });
	Matrix3D<float> trainOutput({ network.getMaximumTrainBatchSize(), 1, 1 });

	int index = 0;
	for (DataSet<float>& dataSet : data) {
		trainInput(index, 0, 0) = dataSet.input(0, 0, 0);
		trainInput(index, 1, 0) = dataSet.input(0, 1, 0);
		trainOutput(index, 0, 0) = dataSet.output(0, 0, 0);
		index++;
	}

	for (int i = 0; i < 4000; i++) {
		network.forward(exec, trainInput);
		network.backward(exec, trainOutput);
		network.update(exec, 0.5f, trainOutput);

		if (i % 100 == 0) {
			std::cout << "Epoch: " << i << "\n";
			std::cout << "Training ";
			timer2.stop();
			timer2.start();

			int index = 0;

			for (int y = 0; y < canvas.getHeight(); y++) {
				for (int x = 0; x < canvas.getWidth(); x++) {

					input(index, 0, 0) = (float)x / canvas.getWidth();
					input(index, 1, 0) = (float)y / canvas.getHeight();

					index++;

					if (index == network.getMaximumBatchSize()) {

						network.forward(exec, input);

						int startIdx = x + y * canvas.getWidth() - network.getMaximumBatchSize() + 1;

						for (int i = 0; i < network.getMaximumBatchSize(); i++) {
							uint8_t color = (uint8_t)(network.getOutput()(i, 0, 0) * 255.0f);
							canvas.setPixel(startIdx + i, color, color, color);
						}

						index = 0;
					}
				}
			}

			canvas.update();
			if (!canvas.frame()) {
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