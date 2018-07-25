#include "simple_screen_listener.h"

namespace deepf1
{

	simple_screen_listener::simple_screen_listener(std::shared_ptr<const boost::timer::cpu_timer> timer,
		cv::Rect2d capture_area,
		std::string application,
		)
	{
		this->timer = timer;
		dataz.reserve(1);
		svc.reset(new screen_video_capture(capture_area, timer, application));
		this->capture_area = svc->capture_area();
		init_images(this->capture_area.height, this->capture_area.width);
	}
	simple_screen_listener::~simple_screen_listener()
	{

	}
	std::vector<timestamped_image_data_t> simple_screen_listener::get_data()
	{
		return dataz;
	}
	void simple_screen_listener::init_images(int num_rows, int num_columns)
	{
		for (unsigned int i = 0; i < 1; i++) {
			timestamped_image_data_t to_add;
			to_add.image.create(num_rows, num_columns, CV_8UC4);
			dataz.push_back(to_add);
		}
	}
	void simple_screen_listener::listen()
	{
		running = true;
		for (unsigned int i = 0; i < dataz.size() && running; i++)
		{
			dataz[i].timestamp = svc->readTimed(dataz[i].image);
			Sleep(3);
		}
	}

}