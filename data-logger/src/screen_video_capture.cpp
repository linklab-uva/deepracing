#include "screen_video_capture.h"

namespace deepf1
{

	screen_video_capture::screen_video_capture(cv::Rect2d capture_area, int displayIndex)
	{
		captureArea = capture_area;
		if (displayIndex >= 0)
			open(displayIndex);
	}

	void screen_video_capture::open(int displayIndex)
	{
		MonitorIndexLookupInfo enumState = { displayIndex, NULL, 0 };
		EnumDisplayMonitors(NULL, NULL, monitorEnumProc, (LPARAM)&enumState);
		this->targetWindow = GetDesktopWindow();

		printf("Capture Area is %f X %f pixels.\n", captureArea.height, captureArea.width);
	}
	cv::Rect2d screen_video_capture::capture_area() const {
		return cv::Rect2d(this->captureArea);
	}
	BOOL CALLBACK screen_video_capture::monitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData)
	{
		MonitorIndexLookupInfo* enumState = (MonitorIndexLookupInfo*)dwData;
		if (enumState->targetIndex == enumState->currentIndex)
		{
			enumState->outRect = *lprcMonitor;
			return false;
		}

		enumState->currentIndex++;

	}

	screen_video_capture::~screen_video_capture()
	{
	}

	void screen_video_capture::read(cv::Mat* destination)
	{
		if (targetWindow == NULL)
			throw new std::exception("No target monitor specified! The 'open()' method must be called to select a target monitor before frames can be read.");

		captureHwnd(targetWindow, captureArea, destination);
	}

	screen_video_capture& screen_video_capture::operator>>(cv::Mat* destination)
	{
		read(destination);
		return *this;
	}

	void screen_video_capture::captureHwnd(HWND window, cv::Rect2d targetArea, cv::Mat* dest)
	{
		HDC hwindowDC, hwindowCompatibleDC;

		HBITMAP hbwindow;
		BITMAPINFOHEADER  bi;

		hwindowDC = GetDC(window);
		hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
		SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);
		//dest.create(targetArea.height, targetArea.width, CV_8UC4);

		// Initialize a bitmap
		hbwindow = CreateCompatibleBitmap(hwindowDC, targetArea.width, targetArea.height);
		bi.biSize = sizeof(BITMAPINFOHEADER);
		bi.biWidth = targetArea.width;
		// The negative height is required -- removing the inversion will make the image appear upside-down.
		bi.biHeight = -targetArea.height;
		bi.biPlanes = 1;
		bi.biBitCount = 32;
		bi.biCompression = BI_RGB;
		bi.biSizeImage = 0;
		bi.biXPelsPerMeter = 0;
		bi.biYPelsPerMeter = 0;
		bi.biClrUsed = 0;
		bi.biClrImportant = 0;

		SelectObject(hwindowCompatibleDC, hbwindow);
		// Copy from the window device context to the bitmap device context
		// Use BitBlt to do a copy without any stretching -- the output is of the same dimensions as the target area.
		BitBlt(hwindowCompatibleDC, 0, 0, targetArea.width, targetArea.height, hwindowDC, targetArea.x, targetArea.y, SRCCOPY);
		// Copy into our own buffer as device-independent bitmap
		GetDIBits(hwindowCompatibleDC, hbwindow, 0, targetArea.height, dest->data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);

		// Clean up memory to avoid leaks
		DeleteObject(hbwindow);
		DeleteDC(hwindowCompatibleDC);
		ReleaseDC(window, hwindowDC);
	}
}