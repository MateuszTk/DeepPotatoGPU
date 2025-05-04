#pragma once

#include "external.hpp"
#include "math/matrix.hpp"

template <typename T>
struct DataSet {
	Matrix3D<T> input;
	Matrix3D<T> output;

	DataSet(const std::initializer_list<T>& input, const std::initializer_list<T>& output, uint32_t batchSize = 1)
		: input({ batchSize, (unsigned int)input.size(), 1 }, input), output({ batchSize, (unsigned int)output.size(), 1 }, output) {

		this->input.getBuffer().setDirection(BufferDirection::HostToDevice);
		this->output.getBuffer().setDirection(BufferDirection::HostToDevice);
	}
};
