#pragma once

#include <iostream>
#include <fstream>

namespace IDX {
	struct IDX_Header {
		unsigned char dataType = 0;
		unsigned char dimensions = 0;
		unsigned int* sizes = nullptr;

		IDX_Header() {}

		IDX_Header(IDX_Header&& other) {
			dataType = other.dataType;
			dimensions = other.dimensions;
			sizes = other.sizes;
			other.sizes = nullptr;
		}

		~IDX_Header() {
			delete[] sizes;
		}
	};

	struct IDX_Data {
		IDX_Header header;
		unsigned char* data = nullptr;
		unsigned int dataSize = 0;

		IDX_Data() {}

		IDX_Data(IDX_Data&& other) : header(std::move(other.header)) {
			data = other.data;
			dataSize = other.dataSize;
			other.data = nullptr;
		}

		~IDX_Data() {
			delete[] data;
		}
	};

	unsigned int readInt(std::ifstream& file) {
		unsigned int value = 0;
		file.read((char*)&value, sizeof(unsigned char) * 4);
		// LSB first to MSB first
		value = ((value & 0x000000FF) << 24) |
			((value & 0x0000FF00) << 8) |
			((value & 0x00FF0000) >> 8) |
			((value & 0xFF000000) >> 24);
		return value;
	}

	IDX_Data import(const char* path) {
		IDX_Data data;
		std::ifstream file(path, std::ios::binary);
		if (!file.is_open()) {
			std::cout << "Error: could not open file: " << path << '\n';
			return data;
		}

		// skip first 2 bytes of magic number
		file.seekg(2, file.beg);

		file.read((char*)&data.header.dataType, sizeof(unsigned char));
		file.read((char*)&data.header.dimensions, sizeof(unsigned char));
		data.header.sizes = new unsigned int[data.header.dimensions];
		for (int i = 0; i < data.header.dimensions; i++) {
			data.header.sizes[i] = readInt(file);
		}

		if (data.header.dataType != 0x08) {
			std::cout << "Error: magic number is not 0x08\n";
			return data;
		}

		unsigned int size = 1;
		for (int i = 0; i < data.header.dimensions; i++) {
			size *= data.header.sizes[i];
		}
		data.dataSize = size;
		data.data = new unsigned char[size];
		file.read((char*)data.data, sizeof(unsigned char) * size);

		file.close();
		return data;
	}

	void printHeader(const IDX_Header& header) {
		if (header.sizes != nullptr) {
			std::cout << "Data type: " << (int)header.dataType << '\n';
			std::cout << "Dimensions: " << (int)header.dimensions << '\n';
			std::cout << "Sizes: ";
			for (int i = 0; i < header.dimensions; i++) {
				std::cout << (int)header.sizes[i] << ' ';
			}
			std::cout << '\n';
		}
	}

	void printData(const IDX_Data& data) {
		printHeader(data.header);
		if (data.data != nullptr) {
			std::cout << "Data: ";
			for (int i = 0; i < 10; i++) {
				std::cout << (int)data.data[i] << ' ';
			}
			std::cout << '\n';
		}
	}
}
