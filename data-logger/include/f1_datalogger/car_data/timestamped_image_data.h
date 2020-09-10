#ifndef TIMESTAMPED_IMAGE_DATA_H
#define  TIMESTAMPED_IMAGE_DATA_H


#include "f1_datalogger/car_data/time_point.h"
#include <opencv2/core.hpp>
namespace deepf1
{
	struct timestamped_image_data {
		public:
			struct timestamped_image_data() = default;
			struct timestamped_image_data(const TimePoint& timestamp, const cv::Mat& image)
			{
				this->image = image;
				this->timestamp = timestamp;
			}
			cv::Mat image;
			TimePoint timestamp;
	};typedef struct timestamped_image_data TimestampedImageData;
}


#endif
