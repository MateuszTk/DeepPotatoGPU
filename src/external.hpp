#pragma once

// C++
#include <initializer_list>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <concepts>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>

// CUDA
#ifdef CUDA_AVAILIABLE
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
#else
    struct uint3 {
        unsigned int x, y, z;
    };

    struct dim3 {
        unsigned int x, y, z;

        dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
        dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
        operator uint3() const {
            return { x, y, z };
        }
    };

    #define __global__
    #define __device__
    #define __host__

    #define CUDAExecutor CPUExecutor
#endif

__host__ __device__ uint3 operator+(const uint3& a, const uint3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__host__ __device__ uint3 operator*(const uint3& a, const uint3& b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

// Options
#define BUFFER_DEBUG_ON 0
#define EXECUTOR_DEBUG_ON 0
