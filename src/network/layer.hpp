#pragma once

#include "external.hpp"
#include "math/matrix.hpp"

enum class Activation {
	Linear,
	ReLU,
	Sigmoid
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

	using WeightsLowMat_t = Matrix2D<lowp_t>;
	using WeightsMat_t = Matrix2D<float>;
	using BiasesMat_t = Matrix1D<float>;
	using OutputsMat_t = Matrix2D<lowp_t>;
	using ErrorsMat_t = Matrix2D<float>;
	using InputsMat_t = Matrix2D<float>;

	WeightsMat_t weights;
	WeightsLowMat_t weightsLow;
	BiasesMat_t biases;
	OutputsMat_t outputs;
	ErrorsMat_t errors;
	InputsMat_t inputs;
	LayerType type;

	Layer(const LayerType& type, uint32_t inputSize, uint32_t batchSize, uint32_t maxTrainBatchSize) :
		weights({ type.getNeurons(), inputSize }),
		weightsLow({ (std::is_same<float, lowp_t>::value ? 0 : type.getNeurons()), inputSize }),
		biases({ type.getNeurons() }),
		outputs({ batchSize, type.getNeurons() }),
		errors({ maxTrainBatchSize, type.getNeurons() }),
		inputs({ batchSize, type.getNeurons() }),
		type(type) {

		outputs.getBuffer().setDirection(BufferDirection::DeviceToHost);

	}
};
