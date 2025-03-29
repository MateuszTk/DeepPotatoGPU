
#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "utils/timer.hpp"

#include "window.hpp"

std::array<DataSet<float>, 4> data = {{
	{{ 0, 0 }, { 0 }},
	{{ 0, 1 }, { 1 }},
	{{ 1, 0 }, { 1 }},
	{{ 1, 1 }, { 0 }}
}};

int main() {

	Window window(400, 400);

	CUDAExecutor exec;

	Timer timer, timer2;

	Network network({
		InputLayer(2),
		DenseLayer(3, Activation::Sigmoid),
		DenseLayer(1, Activation::Sigmoid)
	}, 400 * 400);

	srand(8888);
	network.initialize();

	Matrix3D<float> input({ network.getMaximumBatchSize(), 2, 1 });

	for (int i = 0; i < 4000; i++) {
		for (DataSet<float>& dataSet : data) {
			network.forward(exec, dataSet.input);
			network.backward(exec, dataSet.output);
			network.update(exec, 0.2f);
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

    return 0;
}

