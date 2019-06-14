#ifndef INCLUDE_CONTROLLERS_VJOY_INTERFACE_H_
#define INCLUDE_CONTROLLERS_VJOY_INTERFACE_H_
#include <cmath>
#include <f1_datalogger/controllers/f1_interface.h>
#include <vJoy++/vjoy.h>
namespace deepf1 {

	class VJoyInterface : public F1Interface
	{
	public:
		VJoyInterface(const unsigned int& device_id = 1);
		virtual ~VJoyInterface();
		void setCommands(const F1ControlCommand& command) override;
	private:
		vjoy_plusplus::vJoy vjoy_;
		double max_vjoysteer_, max_vjoythrottle_ , max_vjoybrake_;
	};
	
}
#endif
