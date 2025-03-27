
#include "external.hpp"

#include "compute/cpu.hpp"
#include "compute/cuda.hpp"

#include "math/matrix.hpp"
#include "network/network.hpp"
#include "network/data.hpp"

#include "utils/timer.hpp"

std::array<DataSet<float>, 4> data = {{
	{{ 0, 0 }, { 0 }},
	{{ 0, 1 }, { 1 }},
	{{ 1, 0 }, { 1 }},
	{{ 1, 1 }, { 0 }}
}};

int main() {

	//test();

	CUDAExecutor exec;

	Timer timer;

	Network network({
		InputLayer(2),
		DenseLayer(3, Activation::Sigmoid),
		DenseLayer(1, Activation::Sigmoid)
	});

	srand(time(NULL));
	network.initialize();

	for (int i = 0; i < 4000; i++) {
		for (DataSet<float>& dataSet : data) {
			network.forward(exec, dataSet.input);
			network.backward(exec, dataSet.output);
			network.update(exec, 0.2f);
		}
	}
	
	
	for (DataSet<float>& dataSet : data) {
		network.forward(exec, dataSet.input);
		std::cout << "Input: " << dataSet.input << " Output: " << network.getOutput() << "\n";
	}

	timer.stop();

    return 0;
}

