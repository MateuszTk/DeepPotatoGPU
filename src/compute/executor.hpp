#pragma once

#include "external.hpp"

#include "generic.hpp"
#include "buffer.hpp"

#define ARGS_TO_STRING(...) ([&](){ return ((std::string(typeid(__VA_ARGS__).name()) + ", ") + ...); }().c_str())

class Executor {

	public:

		Executor() = default;
		virtual ~Executor() = default;

		virtual void synchronize() = 0;

};
