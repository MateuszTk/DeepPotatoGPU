#pragma once

#include <cstdint>

#ifdef _WIN32
	#include <windows.h>
#else
	#include <X11/Xlib.h>
	#include <X11/Xutil.h>
#endif

class Window {

	private:

		#ifdef _WIN32
			HWND hwnd;
			HDC hdc;
			BITMAPINFO bmi{};
		#else

		#endif

		int width;
		int height;
		uint8_t* pixels;

	public:

		Window(int width, int height);
		~Window();

		void update();
		void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);
		void setPixel(int index, uint8_t r, uint8_t g, uint8_t b);
		bool frame();

		int getWidth() const;
		int getHeight() const;

};
