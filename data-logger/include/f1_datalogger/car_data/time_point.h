#ifndef F1_DATALOGGER_TIMEPOINT_H
#define F1_DATALOGGER_TIMEPOINT_H
#include <chrono>
#include <memory>
namespace deepf1
{

	typedef std::chrono::high_resolution_clock Clock;
	typedef Clock::time_point TimePoint;
	typedef std::shared_ptr<Clock> ClockPtr;

}
#endif
