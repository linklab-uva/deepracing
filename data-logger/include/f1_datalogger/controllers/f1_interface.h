#ifndef INCLUDE_CONTROLLERS_F1_INTERFACE_H_
#define INCLUDE_CONTROLLERS_F1_INTERFACE_H_
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
	};
	class F1Interface
	{
	public:
		virtual void setCommands(const F1ControlCommand& command) = 0;
	};
}
#endif
