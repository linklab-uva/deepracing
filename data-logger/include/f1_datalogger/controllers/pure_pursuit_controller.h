#include "f1_datalogger/udp_logging/common/measurement_handler.h"
namespace deepf1
{
	class PurePursuitController
	{
	public:
		PurePursuitController(std::shared_ptr<MeasurementHandler> measurement_handler, double Kv=1.0);
		~PurePursuitController();
		void run();
	private:
		std::shared_ptr<MeasurementHandler> measurement_handler_;
	};
}