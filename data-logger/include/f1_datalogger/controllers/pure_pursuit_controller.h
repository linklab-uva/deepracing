#ifndef INCLUDE_CONTROLLERS_PURE_PURSUIT_CONTROLLER_H_
#define INCLUDE_CONTROLLERS_PURE_PURSUIT_CONTROLLER_H_
#include "f1_datalogger/udp_logging/common/measurement_handler.h"
#include "Eigen/Core"
#include <vector>
#include <boost/math/constants/constants.hpp>
namespace deepf1
{
	class PurePursuitController
	{
	public:

		PurePursuitController(std::shared_ptr<MeasurementHandler> measurement_handler, 
			double Kv=1.0, double L = 3.7, double max_angle = 1.0, double throttle = 0.25);
		~PurePursuitController();
		void run();
		static std::vector<Eigen::Vector3d> loadTrackFile(const std::string& filename);

	private:
		std::shared_ptr<MeasurementHandler> measurement_handler_;
		double Kv_, L_,  max_angle_, throttle_;
	};
}
#endif