#pragma once
#include <boost/timer/timer.hpp>
#include <opencv2/core.hpp>
namespace deepf1
{
	struct timestamped_image_data {
		cv::Mat image;
		boost::timer::cpu_times timestamp;
	};typedef struct timestamped_image_data timestamped_image_data_t;
}