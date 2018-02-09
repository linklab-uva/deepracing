#pragma once
#include "car_data/timestamped_image_data.h"
#include <boost/shared_ptr.hpp>
#include "screen_video_capture.h"
#include <vector>
namespace deepf1
{
	
	class simple_screen_listener
	{
	public:
		simple_screen_listener(boost::shared_ptr<const boost::timer::cpu_timer>& timer,
			cv::Rect2d capture_area,
			unsigned int monitor_number = 1,
			unsigned int length = 10);
		~simple_screen_listener();
		void init_images(int num_rows, int num_columns);
		void listen();
		std::vector<timestamped_image_data_t> get_data();
	private:
		unsigned int length;
		boost::shared_ptr<const boost::timer::cpu_timer> timer;
		std::vector<timestamped_image_data_t> dataz;
		boost::shared_ptr<screen_video_capture> svc;
		cv::Rect2d capture_area;
	};

}
