#pragma once

#include "external.hpp"
#include "math/matrix.hpp"

template <typename T>
struct DataSet {
	Matrix2D<T> input;
	Matrix2D<T> output;

	DataSet(const std::initializer_list<T>& input, const std::initializer_list<T>& output) 
		: input({ (unsigned int)input.size(), 1 }, input), output({ (unsigned int)output.size(), 1 }, output) {

	}
};
