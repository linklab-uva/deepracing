
#ifndef INCLUDE_UDP_LOGGING_COMMON_2018_REBROADCAST_HANDLER_H_
#define INCLUDE_UDP_LOGGING_COMMON_2018_REBROADCAST_HANDLER_H_
#include "f1_datalogger/proto_dll_macro.h"
#include "f1_datalogger/udp_logging/visibility_control.h"
#include "f1_datalogger/udp_logging/f1_2018_datagrab_handler.h"
#include <boost/asio.hpp>
#include <thread>
#include <memory>
namespace deepf1
{
		class F1_DATALOGGER_UDP_LOGGING_PUBLIC RebroadcastHandler2018 : public IF12018DataGrabHandler
		{
		public:
			RebroadcastHandler2018();
			virtual ~RebroadcastHandler2018();
			void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data) override;
			void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data) override;
			void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data) override;
			void handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data) override;
			void handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data) override;
			void handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data) override;
			void handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data) override;
			void handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) override;
			inline bool isReady() override;
			void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override;
		private:
  			std::shared_ptr<boost::asio::io_service> io_service_;
			std::shared_ptr<boost::asio::ip::udp::socket> socket_;
			std::shared_ptr<boost::asio::ip::udp::endpoint> remote_endpoint_;
		};
}
#endif