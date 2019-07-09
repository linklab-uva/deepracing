/*
 * multi_threaded_capture.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */
#include "f1_datalogger/f1_datalogger.h"
#include "f1_datalogger/image_logging/common/multi_threaded_framegrab_handler.h"
#include "f1_datalogger/udp_logging/common/multi_threaded_udp_handler.h"
#include "f1_datalogger/udp_logging/common/multi_threaded_udp_handler_2018.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <boost/program_options.hpp>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <GamePad.h>
namespace scl = SL::Screen_Capture;
namespace po = boost::program_options;
void exit_with_help(po::options_description& desc)
{
        std::stringstream ss;
        ss << desc << std::endl;
        std::printf("%s", ss.str().c_str());
        exit(0); // @suppress("Invalid arguments")
}
int main(int argc, char** argv)
{
  using namespace deepf1;
  std::string search_string, image_folder, image_extension, udp_folder, config_file, driver_name, track_name;
  unsigned int image_threads, udp_threads, udp_port;
  float image_capture_frequency, initial_delay_time;


  po::options_description desc("F1 Datalogger Multithreaded Capture. Command line arguments are as follows");
  try{
    desc.add_options()
      ("help,h", "Displays options and exits")
      ("config_file,f", po::value<std::string>(&config_file)->required(), "Configuration file to load")
      ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.find("help") != vm.end()) {
      exit_with_help(desc);
    }
  }catch(boost::exception& e)
  {
    exit_with_help(desc);
  }
  /*
	  search_string: "2017"
	  track_name: "Australia"
	  driver_name: "Trent"
	  images_folder: "images"
	  udp_folder: "udp_data"
	  udp_threads: 3
	  image_threads: 3
	  capture_frequency: 60.0
	  initial_delay_time: 5.0
  */
  std::cout << "Loading config file" << std::endl;
  YAML::Node config_node = YAML::LoadFile(config_file);
  std::cout << "Using the following config information:" << std::endl << config_node << std::endl;

  search_string = config_node["search_string"].as<std::string>();
  image_folder = config_node["images_folder"].as<std::string>();
  udp_folder = config_node["udp_folder"].as<std::string>();
  image_extension = config_node["image_extension"].as<std::string>("jpg");
  udp_threads = config_node["udp_threads"].as<unsigned int>();
  image_threads = config_node["image_threads"].as<unsigned int>();
  udp_port = config_node["udp_port"].as<unsigned int>(20777);
  image_capture_frequency = config_node["image_capture_frequency"].as<float>();
  initial_delay_time = config_node["initial_delay_time"].as<float>();
  /**/




  std::cout<<"Creating handlers" <<std::endl;
  std::shared_ptr<deepf1::MultiThreadedFrameGrabHandler> frame_handler(new deepf1::MultiThreadedFrameGrabHandler(image_extension, image_folder, image_threads, true));
  std::shared_ptr<deepf1::MultiThreadedUDPHandler2018> udp_handler(new deepf1::MultiThreadedUDPHandler2018(udp_folder, true));
  udp_handler->addPausedFunction(std::bind(&deepf1::MultiThreadedFrameGrabHandler::pause, frame_handler.get()));
  std::cout << "Created handlers" << std::endl;


  std::cout<<"Creating DataLogger" <<std::endl;
  std::shared_ptr<deepf1::F1DataLogger> dl( new deepf1::F1DataLogger( search_string, "127.0.0.1", udp_port, false) );
  std::cout<<"Created DataLogger" <<std::endl;


  std::string inp;
  std::cout<<"Enter anything to start capture" << std::endl;
  std::cin >> inp;
	std::cout << "Starting capture in " << initial_delay_time << " seconds." << std::endl;
	std::this_thread::sleep_for(std::chrono::microseconds((long)std::round(initial_delay_time*1E6)));
	dl->start(image_capture_frequency, udp_handler  , frame_handler );
  unsigned int ycount = 0;
  unsigned int bcount = 0;
  std::cout << "Recording. Push Y to pause. B to unpause. D-Pad Down to Exit." << std::endl;
  DirectX::GamePad gp;
  while (true)
  {
    DirectX::GamePad::State gpstate = gp.GetState(0);
    if (gpstate.IsYPressed() || gpstate.IsStartPressed())
    {
      printf("Y is pressed. Pausing %u\n", ++ycount);
      frame_handler->pause();
    }
    if (gpstate.IsStartPressed())
    {
      printf("Start is pressed. Pausing %u\n", ++ycount);
      frame_handler->pause();
    }
    if (gpstate.IsLeftStickPressed())
    {
      printf("Left thumbstick is pressed pressed. Unpausing %u\n", ++bcount);
      frame_handler->resume();
    }
    if (gpstate.IsDPadDownPressed())
    {
      printf("%s","DPad Down is pressed. Exiting\n");
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }

	frame_handler->stop();
	udp_handler->stop();
	frame_handler->join();
	udp_handler->join();
  

}



