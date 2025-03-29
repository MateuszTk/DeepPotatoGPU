
#include "window.hpp"

Window::Window(int width, int height) : width(width), height(height) {
	pixels = new uint8_t[width * height * 4];

	#ifdef _WIN32
		WNDCLASS wc = {};
		wc.lpfnWndProc = DefWindowProc;
		wc.hInstance = GetModuleHandle(0);
		wc.lpszClassName = "Window";

		RegisterClass(&wc);

		hwnd = CreateWindow(wc.lpszClassName, "Window", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, width, height, 0, 0, GetModuleHandle(0), 0);

		ShowWindow(hwnd, SW_SHOW);

		hdc = GetDC(hwnd);

		bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		bmi.bmiHeader.biWidth = width;
		bmi.bmiHeader.biHeight = -height;
		bmi.bmiHeader.biPlanes = 1;
		bmi.bmiHeader.biBitCount = 32;
		bmi.bmiHeader.biCompression = BI_RGB;
	#else

	#endif
}

Window::~Window() {
	delete[] pixels;

	#ifdef _WIN32
		ReleaseDC(hwnd, hdc);
		DestroyWindow(hwnd);
	#else

	#endif
}

void Window::update() {
	#ifdef _WIN32
		StretchDIBits(hdc, 0, 0, width, height, 0, 0, width, height, pixels, &bmi, DIB_RGB_COLORS, SRCCOPY);
	#else

	#endif
}

void Window::setPixel(int index, uint8_t r, uint8_t g, uint8_t b) {
	if (index >= 0 && index < width * height) {
		pixels[index * 4 + 0] = b;
		pixels[index * 4 + 1] = g;
		pixels[index * 4 + 2] = r;
		pixels[index * 4 + 3] = 255;
	}
}

void Window::setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		int index = (y * width + x);
		setPixel(index, r, g, b);
	}
}

bool Window::frame() {
	#ifdef _WIN32
		MSG msg;

		while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		return msg.message != WM_QUIT;
	#else
		return true;
	#endif
}

int Window::getWidth() const {
	return width;
}

int Window::getHeight() const {
	return height;
}
