#include "screen_video_capture.h"

namespace deepf1
{
	screen_video_capture::~screen_video_capture()
	{

		// Clean up memory to avoid leaks
		DeleteObject(hbwindow);
		DeleteDC(hwindowCompatibleDC);
		ReleaseDC(targetWindow, hwindowDC);
	}
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
		targetWindow = GetDesktopWindow();
		hwindowDC = GetDC(targetWindow);
		hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
		SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);
		hbwindow = CreateCompatibleBitmap(hwindowDC, capture_area.width, capture_area.height);
		hwindowDC = GetDC(targetWindow);
		SelectObject(hwindowCompatibleDC, hbwindow); 
		bi.biSize = sizeof(BITMAPINFOHEADER);
		bi.biWidth = capture_area.width;
		// The negative height is required -- removing the inversion will make the image appear upside-down.
		bi.biHeight = -capture_area.height;
		bi.biPlanes = 1;
		bi.biBitCount = 32;
		bi.biCompression = BI_RGB;
		bi.biSizeImage = 0;
		bi.biXPelsPerMeter = 0;
		bi.biYPelsPerMeter = 0;
		bi.biClrUsed = 0;
		bi.biClrImportant = 0;

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


	boost::timer::cpu_times screen_video_capture::read(cv::Mat& destination)
	{
		if (targetWindow == NULL)
			throw new std::exception("No target monitor specified! The 'open()' method must be called to select a target monitor before frames can be read.");

		 return captureHwnd(destination);
	}


	boost::timer::cpu_times screen_video_capture::captureHwnd(cv::Mat& dest)
	{
		
		boost::timer::cpu_times times2 = timer->elapsed();
		BitBlt(hwindowCompatibleDC, 0, 0, captureArea.width, captureArea.height, hwindowDC, captureArea.x, captureArea.y, SRCCOPY);
		// Copy into our own buffer as device-independent bitmap
		GetDIBits(hwindowCompatibleDC, hbwindow, 0, captureArea.height, dest.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);

		return times2;
	}
}