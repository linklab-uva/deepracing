/*
 * multi_threaded_capture.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */
#include "f1_datalogger/f1_datalogger.h"
#include "f1_datalogger/image_logging/common/multi_threaded_framegrab_handler.h"
#include "f1_datalogger/udp_logging/common/multi_threaded_udp_handler.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <boost/program_options.hpp>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <chrono>
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
  std::string search_string, image_folder, udp_folder, config_file, driver_name, track_name;
  unsigned int image_threads, udp_threads;
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
  driver_name = config_node["driver_name"].as<std::string>();
  track_name = config_node["track_name"].as<std::string>();
  udp_threads = config_node["udp_threads"].as<unsigned int>();
  image_threads = config_node["image_threads"].as<unsigned int>();
  image_capture_frequency = config_node["image_capture_frequency"].as<float>();
  initial_delay_time = config_node["initial_delay_time"].as<float>();
  /**/




  std::cout<<"Creating handlers" <<std::endl;
  std::shared_ptr<deepf1::MultiThreadedFrameGrabHandler> frame_handler(new deepf1::MultiThreadedFrameGrabHandler(image_folder, image_threads, true));
  std::shared_ptr<deepf1::MultiThreadedUDPHandler> udp_handler(new deepf1::MultiThreadedUDPHandler(udp_folder, udp_threads, true));


  std::cout<<"Creating DataLogger" <<std::endl;
  std::shared_ptr<deepf1::F1DataLogger> dl( new deepf1::F1DataLogger( search_string , frame_handler , udp_handler ) );
  std::cout<<"Created DataLogger" <<std::endl;
  std::string inp;
  std::cout<<"Enter anything to start capture" << std::endl;
  std::cin >> inp;

  std::cout << "Starting capture in " << initial_delay_time << " seconds." << std::endl;
  std::this_thread::sleep_for(std::chrono::microseconds((long)std::round(initial_delay_time*1E6)));
  dl->start(image_capture_frequency);

  std::cout<<"Capturing data. Enter any key to end " << std::endl;
  std::cin >> inp;

  frame_handler->stop();
  udp_handler->stop();
  frame_handler->join();
  udp_handler->join();

}



