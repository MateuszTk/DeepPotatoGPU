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
		return 0.0f;
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
		return 0.0f;
		#endif
	}

	void resetEnergyConsumption() override {
		#ifdef CUDA_AVAILABLE
		lastEnergy = getTotalEnergyConsumption();
		#endif
	}

	float getEnergyConsumption() {
		#ifdef CUDA_AVAILABLE
		float currentEnergy = getTotalEnergyConsumption();
		float energyConsumed = currentEnergy - lastEnergy;
		return energyConsumed;
		#else
		return 0.0f;
		#endif
	}
};

class CPUStats : public Stats {
private:
	Timer timer;
	float energy = 0.0f;
	float power = 0.0f;
	float startEnergy = 0.0f;
	const std::string linuxPath = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
	std::string windowsPath = ".";
	std::ifstream file;
	int colIndex = -1;
	std::string lastLine;
	int columns = 0;

	bool isLineComplete(std::string line) {
		int commaCount = std::count(line.begin(), line.end(), ',');
		return commaCount + 1 == columns;
	}

	float readColumn(std::string line) {
		int pos = 0;
		int currentIndex = 0;
		while ((pos = line.find(',', 0)) != std::string::npos) {
			if (currentIndex == colIndex) {
				std::string valueStr = line.substr(0, pos);
				try {
					float value = std::stof(valueStr);
					return value;
				}
				catch (const std::invalid_argument&) {
					throw std::runtime_error("Invalid power value in log file");
				}
			}
			line = line.substr(pos + 1);
			currentIndex++;
		}

		throw std::runtime_error("Column index out of range in log file");
	}

	float readRAPL() {
		std::ifstream energyFile(linuxPath);
		if (!energyFile.is_open()) {
			throw std::runtime_error("Failed to open RAPL energy file");
		}
		unsigned long long energyMicroJoules;
		energyFile >> energyMicroJoules;
		energyFile.close();
		float energyWh = static_cast<float>(energyMicroJoules) / (3600.0f * 1000000.0f);
		return energyWh;
	}

public:
	CPUStats() {
		resetEnergyConsumption();
		timer.start();
	}

	~CPUStats() override {
		if (file.is_open()) {
			file.close();
		}
	}

	void setMeasurementFile(const std::string& path) {
		windowsPath = path;
	}

	float getPowerUsage() override {
		#ifdef _WIN32
		if (!file.is_open()){
			file.open(windowsPath);
			if (!file.is_open()) {
				throw std::runtime_error("Failed to open power log file");
			}

			std::string header;
			std::getline(file, header);

			columns = std::count(header.begin(), header.end(), ',') + 1;

			const std::string column = "\"CPU Package Power [W]\"";
			colIndex = -1;
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
		}

		std::string line;
		while (std::getline(file, line)) {
			if (isLineComplete(line)) {
				lastLine = line;
			}
			else {
				// move back to the beginning of the incomplete line
				file.seekg(-static_cast<int>(line.length()), std::ios::cur);
				break;
			}
		}

		file.clear();

		if (lastLine.empty()) {
			throw std::runtime_error("Power log file is empty");
		}
		
		return readColumn(lastLine);
		#else
		return power;
		#endif
	}

	void resetEnergyConsumption() override {
		energy = 0.0f;
		startEnergy = readRAPL();
		power = 0.0f;
		timer.start();
	}

	void tick() override {
		#ifdef _WIN32
		float timeElapsed = timer.stop(false);
		float currPower = getPowerUsage();
		energy += currPower * (timeElapsed / 3600.0f);
		timer.start();
		#else
		float timeElapsed = timer.stop(false);
		float energyWh = readRAPL();
		energy = energyWh - startEnergy;
		power = energy / (timeElapsed / 3600.0f);
		#endif
	}

	float getEnergyConsumption() override {
		return energy;
	}
};
