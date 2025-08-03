#pragma once

#include "external.hpp"
#include "math/matrix.hpp"

template <typename T>
struct DataSet {
	Matrix2D<T> input;
	Matrix2D<T> output;

	DataSet(const std::initializer_list<T>& input, const std::initializer_list<T>& output, uint32_t batchSize = 1)
		: input({ batchSize, (unsigned int)input.size() }, input), output({ batchSize, (unsigned int)output.size()}, output) {

		this->input.getBuffer().setDirection(BufferDirection::HostToDevice);
		this->output.getBuffer().setDirection(BufferDirection::HostToDevice);
	}

	DataSet(uint32_t inputSize, uint32_t outputSize, uint32_t batchSize = 1)
		: input({ batchSize, inputSize }), output({ batchSize, outputSize }) {

		this->input.getBuffer().setDirection(BufferDirection::HostToDevice);
		this->output.getBuffer().setDirection(BufferDirection::HostToDevice);
	}
};
