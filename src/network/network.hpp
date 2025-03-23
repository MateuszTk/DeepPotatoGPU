#pragma once

#include "external.hpp"
#include "layer.hpp"

class Network {

	private:

		std::vector<Layer> layers;

		void initRandom(Matrix2D<float>& matrix) {
			for (unsigned int y = 0; y < matrix.shape(0); y++) {
				for (unsigned int x = 0; x < matrix.shape(1); x++) {
					matrix(y, x) = (float)rand() / RAND_MAX;
				}
			}
		}

	public:

		Network(std::initializer_list<LayerType> layerTypes) {
			layers.reserve(layerTypes.size());

			uint32_t inputSize = 0;

			for (const LayerType& layerType : layerTypes) {
				layers.emplace_back(layerType, inputSize);
				inputSize = layerType.getNeurons();
			}
		}

		virtual ~Network() = default;

		void initialize() {
			for (Layer& layer : layers) {
				initRandom(layer.weights);
				initRandom(layer.biases);
			}
		}

		template <typename Exe>
		void forward(Exe& executor, const Matrix2D<float>& input) {
			Matrix2D<float> currentInput = input;
			
			for (int i = 1; i < layers.size(); i++) {
				Layer& layer = layers[i];

				Matrix2D<float>::multiply(executor, layer.weights, currentInput, layer.outputs);
				Matrix1D<float>::add(executor, layer.outputs, layer.biases, layer.outputs);

				currentInput = layer.outputs;
			}
		}

		Matrix2D<float>& getOutput() {
			return layers.back().outputs;
		}
};
