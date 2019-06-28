#ifndef INCLUDE_CONTROLLERS_PURE_PURSUIT_CONTROLLER_H_
#define INCLUDE_CONTROLLERS_PURE_PURSUIT_CONTROLLER_H_
#include "f1_datalogger/udp_logging/common/measurement_handler_2018.h"
#include "Eigen/Core"
#include <vector>
#include <boost/math/constants/constants.hpp>
#include "f1_datalogger/controllers/f1_interface.h"
#include <memory>
namespace deepf1
{
	class PurePursuitController
	{
	public:

		PurePursuitController(std::shared_ptr<MeasurementHandler2018> measurement_handler, 
			double Kv=1.0, double L = 3.7, double max_angle = 0.78539816339744830961566084581988, double velocity_setpoint = 100.0);
		~PurePursuitController();
		void run(const std::string& trackfile = "Australia_racingline.track", float velKp=1.0, float velKi= 0.1, float velKd = -0.1, float velocity_lookahead_gain = 1.5);

	private:
		std::shared_ptr<MeasurementHandler2018> measurement_handler_;
		double Kv_, L_,  max_angle_, velocity_setpoint_;
		std::shared_ptr<deepf1::F1Interface> f1_interface_;

	};
}
#endif