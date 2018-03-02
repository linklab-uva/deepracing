#include "screen_video_capture.h"

namespace deepf1
{

	screen_video_capture::screen_video_capture(cv::Rect2d capture_area, std::shared_ptr<const boost::timer::cpu_timer> timer, int displayIndex)
	{
		this->timer = std::shared_ptr<const boost::timer::cpu_timer>(timer);
		//captureArea = capture_area;
		if (displayIndex >= 0)
			open(displayIndex, capture_area);
	}

	void screen_video_capture::open(int displayIndex, cv::Rect2d capture_area)
	{
		MonitorIndexLookupInfo enumState = { displayIndex, NULL, 0 };
		EnumDisplayMonitors(NULL, NULL, monitorEnumProc, (LPARAM)&enumState);
		this->captureArea = cv::Rect2d(enumState.outRect.left + capture_area.x, enumState.outRect.top + capture_area.y,
			capture_area.width, capture_area.height);
			//(enumState.outRect.right ) - (enumState.outRect.left ), (enumState.outRect.bottom ) - (enumState.outRect.top ));
		this->targetWindow = GetDesktopWindow();
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

	boost::timer::cpu_times screen_video_capture::read(cv::Mat& destination)
	{
		if (targetWindow == NULL)
			throw new std::exception("No target monitor specified! The 'open()' method must be called to select a target monitor before frames can be read.");

		 return captureHwnd(targetWindow, captureArea, destination);
	}


	boost::timer::cpu_times screen_video_capture::captureHwnd(HWND window, cv::Rect2d targetArea, cv::Mat& dest)
	{
		HDC hwindowDC, hwindowCompatibleDC;

		HBITMAP hbwindow;
		BITMAPINFOHEADER  bi;
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
		hwindowDC = GetDC(window);
		hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
		SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);
		hbwindow = CreateCompatibleBitmap(hwindowDC, targetArea.width, targetArea.height);
		ReleaseDC(window, hwindowDC);


		hwindowDC = GetDC(window);
		SelectObject(hwindowCompatibleDC, hbwindow);
		boost::timer::cpu_times times2 = timer->elapsed();
		BitBlt(hwindowCompatibleDC, 0, 0, targetArea.width, targetArea.height, hwindowDC, targetArea.x, targetArea.y, SRCCOPY);
		// Copy into our own buffer as device-independent bitmap
		GetDIBits(hwindowCompatibleDC, hbwindow, 0, targetArea.height, dest.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);

		// Clean up memory to avoid leaks
		DeleteObject(hbwindow);
		DeleteDC(hwindowCompatibleDC);
		ReleaseDC(window, hwindowDC);
		return times2;
	}
}