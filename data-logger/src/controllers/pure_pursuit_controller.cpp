#include "f1_datalogger/controllers/pure_pursuit_controller.h"



deepf1::PurePursuitController::PurePursuitController(std::shared_ptr<MeasurementHandler> measurement_handler, double Kv)
{
	measurement_handler_ = measurement_handler;
}


deepf1::PurePursuitController::~PurePursuitController()
{
}
