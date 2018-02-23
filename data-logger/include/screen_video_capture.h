#pragma once

#include <opencv2/opencv.hpp>
#include <Windows.h>

namespace deepf1
{
	class screen_video_capture
	{
	public:
		screen_video_capture(cv::Rect2d capture_area, int displayIndex = -1);
		~screen_video_capture();

		void open(int displayIndex, cv::Rect2d capture_area);
		void read(cv::Mat* destination);
		cv::Rect2d capture_area() const;
		screen_video_capture& operator>>(cv::Mat* destination);

	private:
		cv::Rect2d captureArea;
		HWND targetWindow = NULL;


		void captureHwnd(HWND window, cv::Rect2d targetArea, cv::Mat* dest);
		static BOOL CALLBACK monitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
	};

	struct MonitorIndexLookupInfo
	{
		int targetIndex;

		RECT outRect;
		int currentIndex;
	};
}