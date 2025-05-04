#pragma once

#include "external.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class Image {
	private:

		int width;
		int height;
		int channels;
		uint8_t* data;

	public:

		Image(const std::string& path) {
			data = stbi_load(path.c_str(), &width, &height, &channels, 0);

			if (!data) {
				throw std::runtime_error("Failed to load image");
			}
		}

		~Image() {
			stbi_image_free(data);
		}

		uint3 getPixel(int x, int y) const {
			uint3 pixel = { 0, 0, 0 };

			if (x >= 0 && x < width && y >= 0 && y < height) {
				int index = (y * width + x) * channels;

				if (channels >= 1) {
					pixel.x = data[index + 0];
				}
				if (channels >= 2) {
					pixel.y = data[index + 1];
				}
				if (channels >= 3) {
					pixel.z = data[index + 2];
				}
			}

			return pixel;
		}

		uint3 getPixel(float x, float y) const {
			return getPixel((int)(x * width), (int)(y * height));
		}

		int getWidth() const {
			return width;
		}

		int getHeight() const {
			return height;
		}

		int getChannels() const {
			return channels;
		}

		uint8_t* getData() const {
			return data;
		}
};
