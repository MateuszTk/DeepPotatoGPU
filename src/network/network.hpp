#pragma once

#include "external.hpp"
#include "layer.hpp"

class Network {

	private:

		__host__ __device__ static float sigmoid(float x) {
			return 1.0f / (1.0f + exp(-x));
		}

		__host__ __device__ static float sigmoidDerivative(float x) {
			float s = sigmoid(x);
			return s * (1.0f - s);
		}

		__host__ __device__ static float deriverate(float input, Activation activation) {
			if (activation == Activation::Sigmoid) {
				return sigmoidDerivative(input);
			}
			else {
				return 1.0f;
			}
		}

	private:

		std::vector<Layer> layers;
		uint32_t maxBatchSize;

		void initRandom(Matrix2D<float>& matrix) {
			for (unsigned int y = 0; y < matrix.shape(0); y++) {
				for (unsigned int x = 0; x < matrix.shape(1); x++) {
					matrix(y, x) = (float)rand() / RAND_MAX;
				}
			}
		}

	public:

		Network(std::initializer_list<LayerType> layerTypes, uint32_t maxBatchSize) : maxBatchSize(maxBatchSize) {
			layers.reserve(layerTypes.size());

			uint32_t inputSize = 0;

			for (const LayerType& layerType : layerTypes) {
				layers.emplace_back(layerType, inputSize, maxBatchSize);
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

		GENERIC_KERNEL(ForwardLayerKernel) {

			__host__ __device__ float activate(float input, Activation activation) {
				if (activation == Activation::Sigmoid) {
					return sigmoid(input);
				}
				else {
					return input;
				}
			}

			GENERIC_KERNEL_ENTRY(Matrix2D<float> weights, Matrix2D<float> biases, Matrix3D<float> inputs, Matrix3D<float> currentInputs, Matrix3D<float> outputs, Activation activation) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= inputs.shape(2) || index.y >= inputs.shape(1) || index.z >= inputs.shape(0)) {
					return;
				}

				float output = (weights.shape(1) > 0) ? 0.0f : currentInputs(index.z, index.y, 0);

				for (unsigned int i = 0; i < weights.shape(1); i++) {
					output += weights(index.y, i) * currentInputs(index.z, i, index.x);
				}

				output += biases(index.y, 0);

				inputs(index.z, index.y, index.x) = output;
				outputs(index.z, index.y, index.x) = activate(output, activation);
			}
		};

		template <typename Exe>
		void forward(Exe& executor, Matrix3D<float>& input) {

			if (input.shape(0) > layers[0].outputs.shape(0)) {
				throw std::invalid_argument("Input batch size must be no greater than the specified maximum network batch size");
			}

			Activation activation = layers[0].type.getActivation();
			executor.template execute<ForwardLayerKernel>({ layers[0].outputs.shape(2), layers[0].outputs.shape(1), input.shape(0) }, layers[0].weights, layers[0].biases, layers[0].inputs, input, layers[0].outputs, activation);
			
			Matrix3D<float> currentInput = layers[0].outputs;
			
			for (int i = 1; i < layers.size(); i++) {
				Layer& layer = layers[i];

				Activation activation = layer.type.getActivation();
				executor.template execute<ForwardLayerKernel>({ layer.outputs.shape(2), layer.outputs.shape(1), input.shape(0) }, layer.weights, layer.biases, layer.inputs, currentInput, layer.outputs, activation);

				currentInput = layer.outputs;
			}
		}

		GENERIC_KERNEL(OutputLayerErrorKernel) {
			GENERIC_KERNEL_ENTRY(Matrix3D<float> target, Matrix3D<float> output, Matrix2D<float> error, Activation activation) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= target.shape(2) || index.y >= target.shape(1) || index.z >= target.shape(0)) {
					return;
				}

				error(index.y, 0) = (target(0, index.y, 0) - output(0, index.y, 0)) * deriverate(output(0, index.y, 0), activation);
			}
		};

		GENERIC_KERNEL(BackwardLayerKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> weights, Matrix2D<float> errors, Matrix2D<float> prevErrors, Matrix3D<float> prevOutputs, Activation activation) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= prevErrors.shape(1) || index.y >= prevErrors.shape(0)) {
					return;
				}

				float sum = 0.0f;

				for (unsigned int i = 0; i < weights.shape(0); i++) {
					sum += weights(i, index.y) * errors(i, 0);
				}

				sum *= deriverate(prevOutputs(0, index.y, 0), activation);

				prevErrors(index.y, 0) = sum;
			}
		};

		template <typename Exe>
		void backward(Exe& executor, Matrix3D<float>& target) {

			Layer& outputLayer = layers.back();

			Activation activation = outputLayer.type.getActivation();
			executor.template execute<OutputLayerErrorKernel>({ outputLayer.errors.shape(1), outputLayer.errors.shape(0) }, target, outputLayer.outputs, outputLayer.errors, activation);

			for (int i = layers.size() - 2; i >= 0; i--) {
				Layer& layer = layers[i];
				Layer& nextLayer = layers[i + 1];

				Activation activation = layer.type.getActivation();
				executor.template execute<BackwardLayerKernel>({ layer.errors.shape(1), layer.errors.shape(0) }, nextLayer.weights, nextLayer.errors, layer.errors, layer.inputs, activation);
			}
		}

		GENERIC_KERNEL(UpdateWeightsKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> weights, Matrix2D<float> errors, Matrix3D<float> prevOutputs, float learningRate) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= weights.shape(1) || index.y >= weights.shape(0)) {
					return;
				}

				weights(index.y, index.x) += learningRate * errors(index.y, 0) * prevOutputs(0, index.x, 0);
			}
		};

		GENERIC_KERNEL(UpdateBiasesKernel) {
			GENERIC_KERNEL_ENTRY(Matrix2D<float> biases, Matrix2D<float> errors, float learningRate) {
				uint3 index = getThreadIdx() + getBlockIdx() * getBlockDim();

				if (index.x >= biases.shape(1) || index.y >= biases.shape(0)) {
					return;
				}

				biases(index.y, 0) += learningRate * errors(index.y, 0);
			}
		};

		template <typename Exe>
		void update(Exe& executor, float learningRate) {
			for (int i = 0; i < layers.size(); i++) {
				Layer& layer = layers[i];

				if (i > 0) {
					Layer& previousLayer = layers[i - 1];
					executor.template execute<UpdateWeightsKernel>({ layer.weights.shape(1), layer.weights.shape(0) }, layer.weights, layer.errors, previousLayer.outputs, learningRate);
				}

				executor.template execute<UpdateBiasesKernel>({ layer.biases.shape(1), layer.biases.shape(0) }, layer.biases, layer.errors, learningRate);
			}
		}

		Matrix3D<float>& getOutput() {
			return layers.back().outputs;
		}

		uint32_t getMaximumBatchSize() const {
			return maxBatchSize;
		}
};
