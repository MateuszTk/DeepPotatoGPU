#pragma once

#include "external.hpp"
#include "layer.hpp"

class Network {

	public:

		GENERIC_KERNEL(MatrixActivationKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> inputs, Matrix2D<float> outputs, Activation activation) {
				uint3 index = getThreadIdx();
				if (activation == Activation::Sigmoid) {
					outputs(index.y, index.x) = sigmoid(inputs(index.y, index.x));
				}
				else {
					outputs(index.y, index.x) = inputs(index.y, index.x);
				}
			}
		};

		GENERIC_KERNEL(MatrixDeriverationKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> inputs, Matrix2D<float> errors, Activation activation) {
				uint3 index = getThreadIdx();
				if (activation == Activation::Sigmoid) {
					errors(index.y, index.x) *= sigmoidDerivative(inputs(index.y, index.x));
				}
				else {
					errors(index.y, index.x) *= 1.0f;
				}
			}
		};

		GENERIC_KERNEL(UpdateWeightsKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> weights, Matrix2D<float> errors, Matrix2D<float> prevOutputs, float learningRate) {
				uint3 index = getThreadIdx();
				weights(index.y, index.x) += learningRate * errors(index.y, 0) * prevOutputs(index.x, 0);
			}
		};

		GENERIC_KERNEL(UpdateBiasesKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> biases, Matrix2D<float> errors, float learningRate) {
				uint3 index = getThreadIdx();
				biases(index.y, 0) += learningRate * errors(index.y, 0);
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

		__host__ __device__ static float sigmoidDerivative(float x) {
			float s = sigmoid(x);
			return s * (1.0f - s);
		}

		template <typename Exe>
		__host__ static void applyActivation(Exe& executor, Matrix2D<float>& inputs, Matrix2D<float>& outputs, Activation activation) {
			executor.template execute<MatrixActivationKernel>({ inputs.shape(1), inputs.shape(0) }, inputs, outputs, activation);
		}

		template <typename Exe>
		__host__ static void applyDerivation(Exe& executor, Matrix2D<float>& inputs, Matrix2D<float>& errors, Activation activation) {
			executor.template execute<MatrixDeriverationKernel>({ inputs.shape(1), inputs.shape(0) }, inputs, errors, activation);
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
		void forward(Exe& executor, Matrix2D<float>& input) {

			Matrix2D<float>::add(executor, input, layers[0].biases, layers[0].outputs);
			
			Matrix2D<float> currentInput = layers[0].outputs;
			
			for (int i = 1; i < layers.size(); i++) {
				Layer& layer = layers[i];

				Matrix2D<float>::multiply(executor, layer.weights, currentInput, layer.inputs);
				Matrix2D<float>::add(executor, layer.inputs, layer.biases, layer.inputs);
				applyActivation(executor, layer.inputs, layer.outputs, layer.type.getActivation());

				currentInput = layer.outputs;
			}
		}

		template <typename Exe>
		void backward(Exe& executor, Matrix2D<float>& target) {

			Layer& outputLayer = layers.back();
			Matrix2D<float>::subtract(executor, target, outputLayer.outputs, outputLayer.errors);
			applyDerivation(executor, outputLayer.outputs, outputLayer.errors, outputLayer.type.getActivation());

			for (int i = layers.size() - 2; i >= 0; i--) {
				Layer& layer = layers[i];
				Layer& nextLayer = layers[i + 1];
				Matrix2D<float>::multiply(executor, nextLayer.weights, nextLayer.errors, layer.errors, true);
				applyDerivation(executor, layer.inputs, layer.errors, layer.type.getActivation());
			}
		}

		template <typename Exe>
		void update(Exe& executor, float learningRate) {
			//std::cout << "\nUpdate network" << std::endl;
			for (int i = 0; i < layers.size(); i++) {
				Layer& layer = layers[i];
				
				/*std::cout << "\nUpdate layer " << i << std::endl;
				std::cout << "Weights: " << layer.weights << std::endl;
				std::cout << "Biases: " << layer.biases << std::endl;
				std::cout << "Errors: " << layer.errors << std::endl;
				std::cout << "Outputs: " << layer.outputs << std::endl;*/

				if (i > 0) {
					Layer& previousLayer = layers[i - 1];
					executor.template execute<UpdateWeightsKernel>({ layer.weights.shape(1), layer.weights.shape(0) }, layer.weights, layer.errors, previousLayer.outputs, learningRate);
				}

				executor.template execute<UpdateBiasesKernel>({ layer.biases.shape(1), layer.biases.shape(0) }, layer.biases, layer.errors, learningRate);

			}
		}

		Matrix2D<float>& getOutput() {
			return layers.back().outputs;
		}
};
