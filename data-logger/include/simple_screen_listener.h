//#pragma once
#include "car_data/car_data.h"
#include <opencv/cv.h>
#include <boost/timer/timer.hpp>
#include <boost/shared_ptr.hpp>
#include "screen_video_capture.h"
namespace deepf1
{
	struct timestamped_image_data {
		cv::Mat* image;
		boost::timer::cpu_times timestamp;
	};
	typedef struct timestamped_image_data timestamped_image_data_t;


	class simple_screen_listener
	{
	public:
		simple_screen_listener(boost::shared_ptr<const boost::timer::cpu_timer>& timer, unsigned int monitor_number = 1, unsigned int length = 10);
		~simple_screen_listener();
		void init_images(int num_rows, int num_columns);
		void listen();
		timestamped_image_data_t* get_data();
	private:
		unsigned int length;
		boost::shared_ptr<const boost::timer::cpu_timer> timer;
		timestamped_image_data_t* dataz;
		boost::shared_ptr<screen_video_capture> svc;
	};

}
