
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

	CPUExecutor exec;

	Timer timer, timer2;

	Network network({
		InputLayer(2),
		DenseLayer(3, Activation::Sigmoid),
		DenseLayer(1, Activation::Sigmoid)
	});

	srand(time(NULL));
	network.initialize();

	Matrix2D<float> input({ 2, 1 });

	for (int i = 0; i < 4000; i++) {
		for (DataSet<float>& dataSet : data) {
			network.forward(exec, dataSet.input);
			network.backward(exec, dataSet.output);
			network.update(exec, 0.2f);
		}

		if (i % 100 == 0) {
			std::cout << "Epoch: " << i << "\n";
			timer2.stop();

			for (int y = 0; y < window.getHeight(); y++) {
				for (int x = 0; x < window.getWidth(); x++) {
					input = { (float)x / window.getWidth(), (float)y / window.getHeight() };
					network.forward(exec, input);

					uint8_t color = (uint8_t)(network.getOutput()(0, 0) * 255.0f);
					window.setPixel(x, y, color, color, color);
				}
			}

			window.update();
			if (!window.frame()) {
				break;
			}

			//std::this_thread::sleep_for(std::chrono::milliseconds(10));

			timer2.start();
		}
	}
	
	
	for (DataSet<float>& dataSet : data) {
		network.forward(exec, dataSet.input);
		std::cout << "Input: " << dataSet.input << " Output: " << network.getOutput() << "\n";
	}

	timer.stop();

    return 0;
}

