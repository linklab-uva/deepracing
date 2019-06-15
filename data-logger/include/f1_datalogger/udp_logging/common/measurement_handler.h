
#ifndef INCLUDE_UDP_LOGGING_COMMON_MEASUREMENT_HANDLER_H_
#define INCLUDE_UDP_LOGGING_COMMON_MEASUREMENT_HANDLER_H_
#include "f1_datalogger/udp_logging/f1_datagrab_handler.h"
namespace deepf1
{
	class MeasurementHandler : public IF1DatagrabHandler
	{
	public:
		MeasurementHandler(unsigned int buffer_capacity=10);
		virtual ~MeasurementHandler();
		void handleData(const deepf1::TimestampedUDPData& data) override;
		inline bool isReady() override;
		void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override;

		deepf1::TimestampedUDPData getData() const;


	private:
		deepf1::TimestampedUDPData data_;
	};
}
#endif