#include <f1_datalogger/controllers/internal/vigem_utils.h>
#include <algorithm>
#include <cmath>

namespace deepf1
{

XINPUT_GAMEPAD toXinput(const F1ControlCommand& f1_interface_command)
{
    XINPUT_GAMEPAD rtn;
    toXinputInplace(f1_interface_command, rtn);
    return rtn;
}
void toXinputInplace(const F1ControlCommand& f1_interface_command, XINPUT_GAMEPAD& rtn)
{
    rtn.sThumbLY=rtn.sThumbRX=rtn.sThumbRY=0;
    //Don't update values that are NaN
    if(!std::isnan<double>(f1_interface_command.steering))
    {
        //short goes from -32,768 to 32,767
        //but left (negative) on the joystick is steering to the left, which is a positive steering angle in our convention.
        double steering = std::clamp<double>(f1_interface_command.steering, -1.0, 1.0);
        if (steering>0.0)
        {
            rtn.sThumbLX = (SHORT)std::round(-32768.0*steering);
        }
        else
        {
            rtn.sThumbLX = (SHORT)std::round(-32767.0*steering);
        }
    }
    if(!std::isnan<double>(f1_interface_command.throttle))
    {
        //BYTE (unsigned char) goes from 0 to 255
        double throttle = std::clamp<double>(f1_interface_command.throttle, -1.0, 1.0);
        rtn.bRightTrigger = (BYTE)std::round(255.0*throttle);
    }
    if(!std::isnan<double>(f1_interface_command.brake))
    {
        double brake = std::clamp<double>(f1_interface_command.brake, -1.0, 1.0);
        rtn.bLeftTrigger = (BYTE)std::round(255.0*brake);
    }
}

}