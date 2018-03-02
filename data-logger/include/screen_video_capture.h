#pragma once
#include "car_data/timestamped_image_data.h"

#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <memory>
namespace deepf1
{
	class screen_video_capture
	{
	public:
		screen_video_capture(cv::Rect2d capture_area, std::shared_ptr<const boost::timer::cpu_timer> timer, int displayIndex = -1);
		~screen_video_capture();

		void open(int displayIndex, cv::Rect2d capture_area);
		boost::timer::cpu_times read(cv::Mat& destination);
		cv::Rect2d capture_area() const;

	private:
		std::shared_ptr<const boost::timer::cpu_timer> timer;
		cv::Rect2d captureArea;
		HWND targetWindow = NULL;
		HDC hwindowDC, hwindowCompatibleDC;
		HBITMAP hbwindow;
		boost::timer::cpu_times captureHwnd(HWND window, cv::Rect2d targetArea, cv::Mat& dest);
		static BOOL CALLBACK monitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
	};

	struct MonitorIndexLookupInfo
	{
		int targetIndex;

		RECT outRect;
		int currentIndex;
	};
}