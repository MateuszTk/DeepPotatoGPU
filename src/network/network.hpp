#pragma once

#include "external.hpp"
#include "layer.hpp"

class Network {

	public:

		GENERIC_KERNEL(MatrixMapKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> matA, Activation activation) {
				uint3 index = getThreadIdx();
				if (activation == Activation::Sigmoid) {
					matA(index.y, index.x) = sigmoid(matA(index.y, index.x));
				}
				else {
					matA(index.y, index.x) = matA(index.y, index.x);
				}
			}
		};

	private:

		std::vector<Layer> layers;

		void initRandom(Matrix2D<float>& matrix) {
			for (unsigned int y = 0; y < matrix.shape(0); y++) {
				for (unsigned int x = 0; x < matrix.shape(1); x++) {
					matrix(y, x) = (float)rand() / RAND_MAX;
				}
			}
		}

		__host__ __device__ static float sigmoid(float x) {
			return 1.0f / (1.0f + exp(-x));
		}

		template <typename Exe>
		__host__ static void applyActivation(Exe& executor, Matrix2D<float>& matA, Activation activation) {
			executor.template execute<MatrixMapKernel>({ matA.shape(1), matA.shape(0) }, matA, activation);
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
				Matrix2D<float>::add(executor, layer.outputs, layer.biases, layer.outputs);
				applyActivation(executor, layer.outputs, layer.type.getActivation());

				currentInput = layer.outputs;
			}
		}

		Matrix2D<float>& getOutput() {
			return layers.back().outputs;
		}
};
