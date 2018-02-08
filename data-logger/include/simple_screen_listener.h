#pragma once
#include <boost/timer/timer.hpp>
#include "car_data/car_data.h"
#include <boost/shared_ptr.hpp>
class simple_screen_listener
{
public: 
	simple_screen_listener(boost::shared_ptr<const boost::timer::cpu_timer>& timer, unsigned int length = 250);
	~simple_screen_listener();
};

