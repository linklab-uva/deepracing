#ifndef INCLUDE_CONTROLLERS_INTERNAL_VIGEM_UTILS_H_
#define INCLUDE_CONTROLLERS_INTERNAL_VIGEM_UTILS_H_

#include <f1_datalogger/controllers/internal/vigem_decl.h>
#include <f1_datalogger/controllers/visibility_control.h>
#include <f1_datalogger/controllers/f1_interface.h>

namespace deepf1
{

XINPUT_GAMEPAD F1_DATALOGGER_CONTROLS_PUBLIC toXinput(const deepf1::F1ControlCommand& f1_interface_command);
void F1_DATALOGGER_CONTROLS_PUBLIC toXinputInplace(const deepf1::F1ControlCommand& f1_interface_command, XINPUT_GAMEPAD& rtn);

}

#endif