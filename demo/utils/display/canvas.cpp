
#include "canvas.hpp"
#include <stdexcept>

Canvas::Canvas(int width, int height) : width(width), height(height) {
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
		display = XOpenDisplay(nullptr);
		if (!display) {
			throw std::runtime_error("Failed to open X display");
		}

		int screen = DefaultScreen(display);
		window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, width, height, 1, BlackPixel(display, screen), WhitePixel(display, screen));
		XSelectInput(display, window, ExposureMask | KeyPressMask);
		XMapWindow(display, window);

		gc = XCreateGC(display, window, 0, nullptr);
		image = XCreateImage(display, DefaultVisual(display, screen), 24, ZPixmap, 0, reinterpret_cast<char*>(pixels), width, height, 32, 0);
	#endif
}

Canvas::~Canvas() {
	#ifdef _WIN32
		delete[] pixels;
		ReleaseDC(hwnd, hdc);
		DestroyWindow(hwnd);
	#else
		XDestroyImage(image);
		XFreeGC(display, gc);
		XDestroyWindow(display, window);
		XCloseDisplay(display);
	#endif
}

void Canvas::update() {
	#ifdef _WIN32
		StretchDIBits(hdc, 0, 0, width, height, 0, 0, width, height, pixels, &bmi, DIB_RGB_COLORS, SRCCOPY);
	#else
		XPutImage(display, window, gc, image, 0, 0, 0, 0, width, height);
		XFlush(display);
	#endif
}

void Canvas::setPixel(int index, uint8_t r, uint8_t g, uint8_t b) {
	if (index >= 0 && index < width * height) {
		pixels[index * 4 + 0] = b;
		pixels[index * 4 + 1] = g;
		pixels[index * 4 + 2] = r;
		pixels[index * 4 + 3] = 255;
	}
}

void Canvas::setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		int index = (y * width + x);
		setPixel(index, r, g, b);
	}
}

void Canvas::setPixel(float x, float y, uint8_t r, uint8_t g, uint8_t b) {
	if (x >= 0 && x < 1 && y >= 0 && y < 1) {
		setPixel((int)(x * width), (int)(y * height), r, g, b);
	}
}

bool Canvas::frame() {
	#ifdef _WIN32
		MSG msg;

		while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		return msg.message != WM_QUIT;
	#else
		XEvent event;
		while (XPending(display)) {
			XNextEvent(display, &event);
			if (event.type == KeyPress) {
				return false;
			}
		}

		return true;
	#endif
}

int Canvas::getWidth() const {
	return width;
}

int Canvas::getHeight() const {
	return height;
}
