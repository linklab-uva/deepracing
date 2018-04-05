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
		screen_video_capture(cv::Rect2d capture_area, std::shared_ptr<const boost::timer::cpu_timer> timer, std::string application);
		~screen_video_capture();

		boost::timer::cpu_times read(cv::Mat& destination);
		cv::Rect2d capture_area() const;

	private:
		void open(std::string application, cv::Rect2d capture_area);
		std::shared_ptr<const boost::timer::cpu_timer> timer;
		cv::Rect2d captureArea;
		HWND targetWindow = NULL;
		HDC hwindowDC, hwindowCompatibleDC;
		HBITMAP hbwindow;
		boost::timer::cpu_times captureHwnd(cv::Mat& dest);
		static BOOL CALLBACK monitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
		BITMAPINFOHEADER  bi;
	};

	struct MonitorIndexLookupInfo
	{
		int targetIndex;

		RECT outRect;
		int currentIndex;
	};

}