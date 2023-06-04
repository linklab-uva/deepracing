#include "f1_datalogger/controllers/vigem_interface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>
#include <thread>
#include <windows.h>
#include <ViGEm/Client.h>

deepf1::VigemInterface::VigemInterface(const unsigned int& device_id) 
{	
	if (device_id==1){
		vigem_target_ = vigem_target_x360_alloc();
	}
	else{
		vigem_target_ = vigem_target_ds4_alloc();
	}
	id_ = 0;
}
deepf1::VigemInterface::~VigemInterface()
{

}
void deepf1::VigemInterface::pushDRS()
{

}
void deepf1::VigemInterface::setCommands(const F1ControlCommand& command)
{

}
