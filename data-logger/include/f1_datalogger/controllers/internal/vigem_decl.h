#ifndef INCLUDE_CONTROLLERS_INTERNAL_VIGEM_DECL_H_
#define INCLUDE_CONTROLLERS_INTERNAL_VIGEM_DECL_H_
#ifdef _MSC_VER
    #include <windows.h>
    #include <Xinput.h>
    #include <ViGEm/Client.h>
#else
    #error "Only Windows 10/11 is supported for the ViGem interface"
#endif

enum VIGEM_DEVICE_TYPE{
    Xbox360=1,
    DualShock4=2
};

#endif