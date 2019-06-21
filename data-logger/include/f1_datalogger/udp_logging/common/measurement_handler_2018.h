
#ifndef INCLUDE_UDP_LOGGING_COMMON_MEASUREMENT_HANDLER_H_
#define INCLUDE_UDP_LOGGING_COMMON_MEASUREMENT_HANDLER_H_
#include "f1_datalogger/udp_logging/f1_2018_datagrab_handler.h"
namespace deepf1
{
		class MeasurementHandler2018 : public IF12018DataGrabHandler
		{
		public:
			MeasurementHandler2018();
			virtual ~MeasurementHandler2018();
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

			deepf1::twenty_eighteen::TimestampedPacketCarSetupData current_setup_data_;
			deepf1::twenty_eighteen::TimestampedPacketCarStatusData current_status_data_;
			deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData current_telemetry_data_;
			deepf1::twenty_eighteen::TimestampedPacketEventData current_event_data_;
			deepf1::twenty_eighteen::TimestampedPacketLapData current_lap_data_;
			deepf1::twenty_eighteen::TimestampedPacketMotionData current_motion_data_;
			deepf1::twenty_eighteen::TimestampedPacketParticipantsData current_participant_data_;
			deepf1::twenty_eighteen::TimestampedPacketSessionData current_session_data_;

		};
}
#endif