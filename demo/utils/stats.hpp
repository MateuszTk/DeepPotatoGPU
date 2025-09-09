#pragma once
#include "external.hpp"
#include <nvml.h>

class Stats {
private:
	nvmlDevice_t device;
public:
	Stats() {
		nvmlReturn_t result = nvmlInit();
		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to initialize NVML");
		}
		result = nvmlDeviceGetHandleByIndex(0, &device);
		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to get handle for device 0");
		}
	}

	~Stats() {
		nvmlReturn_t result = nvmlShutdown();
		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to shutdown NVML");
		}
	}

	float getPowerUsage() {
		unsigned int power;
		nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);
		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to get power usage");
		}
		return static_cast<float>(power) / 1000.0f;
	}
};
