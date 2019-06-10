#include "f1_datalogger/udp_logging/f1_datagrab_handler.h"
namespace deepf1
{
class MeasurementHandler : public IF1DatagrabHandler
{
	public:
		MeasurementHandler::MeasurementHandler();
		virtual MeasurementHandler::~MeasurementHandler();
		void handleData(const deepf1::TimestampedUDPData& data) override;
		inline bool isReady() override;
		void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override;

		deepf1::TimestampedUDPData getData() const;

	private:
		deepf1::TimestampedUDPData data_;
};
}