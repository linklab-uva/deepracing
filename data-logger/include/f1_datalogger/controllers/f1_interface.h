#ifndef INCLUDE_CONTROLLERS_F1_INTERFACE_H_
#define INCLUDE_CONTROLLERS_F1_INTERFACE_H_

#include <f1_datalogger/visibility_control.h>

namespace deepf1 {
	struct F1ControlCommand
	{
	public:
		double steering;
		double throttle;
		double brake;
		F1ControlCommand()
		{
			steering = 0.0;
			throttle = 0.0;
			brake = 0.0;
		}
		F1ControlCommand(double steering_,double throttle_,double brake_) :
			steering(steering_), throttle(throttle_), brake(brake_)
		{
		}
	};
	class F1_DATALOGGER_PUBLIC F1Interface
	{
	public:
		virtual void setCommands(const F1ControlCommand& command) = 0;
		virtual void pushDRS() = 0;
	};
}
#endif
