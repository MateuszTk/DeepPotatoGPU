#pragma once

#include "external.hpp"
#include "math/matrix.hpp"

enum class Activation {
	Linear,
	ReLU,
	Sigmoid,
	Softmax
};

struct LayerType {

	private:

		uint32_t neurons;
		Activation activation;

	public:

		LayerType(uint32_t neurons, Activation activation) : neurons(neurons), activation(activation) {}
		virtual ~LayerType() = default;

		uint32_t getNeurons() const {
			return neurons;
		}

		Activation getActivation() const {
			return activation;
		}
};

struct InputLayer : public LayerType {
	InputLayer(uint32_t neurons) : LayerType(neurons, Activation::Linear) {}
};

struct DenseLayer : public LayerType {
	DenseLayer(uint32_t neurons, Activation activation) : LayerType(neurons, activation) {}
};

struct Layer {

		Matrix2D<float> weights;
		Matrix2D<float> biases;
		Matrix2D<float> outputs;
		LayerType type;

		Layer(const LayerType& type, uint32_t inputSize) 
			: weights({ type.getNeurons(), inputSize }), biases({ type.getNeurons(), 1 }), outputs({ type.getNeurons(), 1 }), type(type) {
	
		}

};
