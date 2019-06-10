#include "f1_datalogger/controllers/pure_pursuit_controller.h"
#include <thread>


deepf1::PurePursuitController::PurePursuitController(std::shared_ptr<MeasurementHandler> measurement_handler, double Kv)
{
	measurement_handler_ = measurement_handler;
}


deepf1::PurePursuitController::~PurePursuitController()
{
}

void deepf1::PurePursuitController::run()
{
	deepf1::TimestampedUDPData data;
	do
	{
		data = measurement_handler_->getData();
		std::printf("Cars XYZ position: %f %f %f\n", data.data.m_x, data.data.m_y, data.data.m_z);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}while (true);
}