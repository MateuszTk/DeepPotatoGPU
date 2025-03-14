#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdexcept>
#include <initializer_list>

#include "generic.hpp"
#include "buffer.hpp"

class Executor {

	public:

		Executor() = default;
		virtual ~Executor() = default;

};
