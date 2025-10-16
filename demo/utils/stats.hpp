#pragma once
#include "external.hpp"
#include "timer.hpp"

#ifdef CUDA_AVAILABLE
#include <nvml.h>
#endif

class Stats {
public:
	virtual ~Stats() = default;
	virtual float getPowerUsage() = 0;
	virtual void resetEnergyConsumption() = 0;
	virtual float getEnergyConsumption() = 0;
	virtual void tick() {};
};

class DummyStats : public Stats {
public:
	DummyStats() {}
	~DummyStats() override = default;
	float getPowerUsage() override { return 0.0f; }
	void resetEnergyConsumption() override {}
	float getEnergyConsumption() override { return 0.0f; }
};

class CUDAStats : public Stats {
private:
	#ifdef CUDA_AVAILABLE
	nvmlDevice_t device;
	float lastEnergy = 0;
	#endif

	float getTotalEnergyConsumption() {
		#ifdef CUDA_AVAILABLE
		unsigned long long energy;
		nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);
		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to get total energy consumption");
		}
		float energyWattHours = static_cast<float>(energy) / (3600.0f * 1000.0f);
		return energyWattHours;
		#else
		throw std::runtime_error("CUDA is not available");
		#endif
	}

public:
	CUDAStats() {
		#ifdef CUDA_AVAILABLE
		nvmlReturn_t result = nvmlInit();
		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to initialize NVML");
		}

		result = nvmlDeviceGetHandleByIndex(0, &device);

		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to get handle for device 0");
		}

		lastEnergy = getTotalEnergyConsumption();
		#else
		throw std::runtime_error("CUDA is not available");
		#endif
	}

	~CUDAStats() {
		#ifdef CUDA_AVAILABLE
		nvmlReturn_t result = nvmlShutdown();
		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to shutdown NVML");
		}
		#endif
	}

	float getPowerUsage() override {
		#ifdef CUDA_AVAILABLE
		unsigned int power;
		nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);
		if (result != NVML_SUCCESS) {
			throw std::runtime_error("Failed to get power usage");
		}
		return static_cast<float>(power) / 1000.0f;
		#else
		throw std::runtime_error("CUDA is not available");
		#endif
	}

	void resetEnergyConsumption() override {
		#ifdef CUDA_AVAILABLE
		lastEnergy = getTotalEnergyConsumption();
		#else
		throw std::runtime_error("CUDA is not available");
		#endif
	}

	float getEnergyConsumption() {
		#ifdef CUDA_AVAILABLE
		float currentEnergy = getTotalEnergyConsumption();
		float energyConsumed = currentEnergy - lastEnergy;
		return energyConsumed;
		#else
		throw std::runtime_error("CUDA is not available");
		#endif
	}
};

class CPUStats : public Stats {
private:
	Timer timer;
	float energy = 0.0f;
	//const std::string linuxPath = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
	const std::string windowsPath = "C:\\Users\\mateu\\Downloads\\hwlog.CSV";

public:
	CPUStats() {
		energy = 0.0f;
		timer.start();
	}

	float getPowerUsage() override {
		static std::ifstream file;
		file.open(windowsPath);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open power log file");
		}

		std::string header;
		std::getline(file, header);
		std::string lastLine;
		std::string line;
		while (std::getline(file, line)) {
			lastLine = line;
		}
		file.close();

		if (lastLine.empty()) {
			throw std::runtime_error("Power log file is empty");
		}

		const std::string column = "\"CPU Package Power [W]\"";
		int colIndex = -1;
		size_t pos = 0;
		int currentIndex = 0;
		while ((pos = header.find(',', 0)) != std::string::npos) {
			std::string col = header.substr(0, pos);
			if (col == column) {
				colIndex = currentIndex;
				break;
			}
			header = header.substr(pos + 1);
			currentIndex++;
		}
		if (colIndex == -1) {
			throw std::runtime_error("Column not found in power log file");
		}
		pos = 0;
		currentIndex = 0;
		while ((pos = lastLine.find(',', 0)) != std::string::npos) {
			if (currentIndex == colIndex) {
				std::string valueStr = lastLine.substr(0, pos);
				try {
					float value = std::stof(valueStr);
					return value;
				}
				catch (const std::invalid_argument&) {
					throw std::runtime_error("Invalid power value in log file");
				}
			}
			lastLine = lastLine.substr(pos + 1);
			currentIndex++;
		}

		throw std::runtime_error("Power value not found in log file");
	}

	void resetEnergyConsumption() override {
		energy = 0.0f;
		timer.start();
	}

	void tick() override {
		float power = getPowerUsage();
		float timeElapsed = timer.stop(false);
		energy += power * (timeElapsed / 3600.0f);
		timer.start();
	}

	float getEnergyConsumption() override {
		return energy;
	}
};
