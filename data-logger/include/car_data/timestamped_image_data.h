#ifndef TIMESTAMPED_IMAGE_DATA_H
#define  TIMESTAMPED_IMAGE_DATA_H


#include <chrono>
#include <opencv2/core.hpp>
namespace deepf1
{
	struct timestamped_image_data {
		cv::Mat image;
		std::chrono::microseconds timestamp;
	};typedef struct timestamped_image_data timestamped_image_data_t;
}


#endif