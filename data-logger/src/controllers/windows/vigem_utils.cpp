#include <f1_datalogger/controllers/internal/vigem_utils.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <iostream>

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
        double xratio = f1_interface_command.steering;
        double xratioabs = std::abs(xratio);
        rtn.sThumbLX = std::round(-32767.0*xratio);
        rtn.sThumbLY = 0;
        // double xangle = xratio*M_PI_2;

        // double sinxangle = std::sin(xangle);
        // double sinxangleabs = std::abs(sinxangle);

        // double asin = std::asin(xratio);
        // if (xratio>0.0)
        // {
        //     // rtn.sThumbLX = (SHORT)std::round(-32765.0*(asin/M_PI_2));
        // }
        // else if(xratio<0.0)
        // {
        //     // rtn.sThumbLX = (SHORT)std::round(32765.0*(-asin/M_PI_2));
        //     // rtn.sThumbLX = std::round(32767.0*sinxangleabs);
        //     rtn.sThumbLX = std::round(-32760.0*xratio);
        // }
        // else
        // {
        //     rtn.sThumbLX = 0;
        // }
        // double yratio = std::sqrt(1.0-(xratio*xratio));
        // double yangle = yratio*M_PI_2;
        // double sinyangle = std::sin(yangle);
        // rtn.sThumbLY = std::round(32765.0*sinyangle);
        // std::cerr<<"rtn.sThumbLX: " << rtn.sThumbLX << std::endl;
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