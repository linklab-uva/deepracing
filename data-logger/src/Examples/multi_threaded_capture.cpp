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
#include "f1_datalogger/udp_logging/common/rebroadcast_handler_2018.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <boost/program_options.hpp>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <map>
#ifdef _MSC_VER
  #if _WIN32_WINNT>=_WIN32_WINNT_WIN10
    #include <wrl/wrappers/corewrappers.h>
    #include <wrl/client.h>
  #endif
#endif
#include "f1_datalogger/filesystem_helper.h"
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
  #ifdef _MSC_VER
    #if _WIN32_WINNT>=_WIN32_WINNT_WIN10
      Microsoft::WRL::Wrappers::RoInitializeWrapper initialize(RO_INIT_MULTITHREADED);
    #endif
  #endif
  using namespace deepf1;
  std::string search_string, image_folder, image_extension, udp_folder, config_file, root_directory, driver_name;
  unsigned int image_threads, udp_port, udp_thread_sleeptime;
  float image_capture_frequency, initial_delay_time;
  bool spectating, use_json, init_paused, log_images, rebroadcast;
  double capture_region_ratio;
  std::map<std::string, unsigned int> udp_thread_dict;


  po::options_description desc("F1 Datalogger Multithreaded Capture. Command line arguments are as follows");
  try{
    desc.add_options()
      ("help,h", "Displays options and exits")
      ("config_file,f", po::value<std::string>(&config_file)->required(), "Configuration file to load")
      ("root_directory,r", po::value<std::string>(&root_directory)->required(), "Root directory to put dataset")
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

  search_string = config_node["search_string"].as<std::string>();
  image_folder = config_node["images_folder"].as<std::string>();
  udp_folder = config_node["udp_folder"].as<std::string>();
  image_extension = config_node["image_extension"].as<std::string>("jpg");
  image_threads = config_node["image_threads"].as<unsigned int>();
  udp_port = config_node["udp_port"].as<unsigned int>(20777);
  udp_thread_sleeptime = config_node["udp_thread_sleeptime"].as<unsigned int>(75);  
  image_capture_frequency = config_node["image_capture_frequency"].as<float>();
  initial_delay_time = config_node["initial_delay_time"].as<float>();
  spectating = config_node["spectating"].as<bool>(false);
  init_paused = config_node["init_paused"].as<bool>(false);
  use_json = config_node["use_json"].as<bool>(true);
  log_images = config_node["log_images"].as<bool>(true);
  rebroadcast = config_node["rebroadcast"].as<bool>(false);
  capture_region_ratio = config_node["capture_region_ratio"].as<double>(1.0);
  udp_thread_dict = config_node["udp_threads"].as<std::map<std::string,unsigned int>>(std::map<std::string,unsigned int>());
  
  
  config_node["search_string"] = search_string;
  config_node["images_folder"] = image_folder;
  config_node["udp_folder"] = udp_folder;
  config_node["image_extension"] = image_extension;
  config_node["image_threads"] = image_threads;
  config_node["udp_thread_sleeptime"] = udp_thread_sleeptime;
  config_node["udp_port"] = udp_port;
  config_node["image_capture_frequency"] = image_capture_frequency;
  config_node["initial_delay_time"] = initial_delay_time;
  config_node["spectating"] = spectating;
  config_node["init_paused"] = init_paused;
  config_node["use_json"] = use_json;
  config_node["capture_region_ratio"] = capture_region_ratio;
  std::cout << "Using the following config information:" << std::endl << config_node << std::endl;
  fs::path root_dir(root_directory);
  if(!fs::is_directory(root_dir))
  {
    fs::create_directories(root_dir);
  }
  std::fstream yamlout;
  yamlout.open((root_dir/fs::path("f1_dataset_config.yaml")).string(), std::fstream::out | std::fstream::trunc);
  yamlout<<config_node;
  yamlout.close();
  


  std::cout<<"Creating handlers" <<std::endl;
  std::shared_ptr<deepf1::MultiThreadedFrameGrabHandler> frame_handler;
  if(log_images)
  {
    deepf1::MultiThreadedFrameGrabHandlerSettings settings;
    settings.image_extension=image_extension;
    settings.images_folder=(root_dir/fs::path(image_folder)).string();
    settings.thread_count=image_threads;
    settings.write_json=use_json;
    settings.capture_region_ratio=capture_region_ratio;
    frame_handler.reset(new deepf1::MultiThreadedFrameGrabHandler(settings));
    frame_handler->resume();
  }
  deepf1::MultiThreadedUDPHandler2018Settings udp_settings;
  udp_settings.write_json=use_json;
  udp_settings.udp_directory=(root_dir/fs::path(udp_folder)).string();
  udp_settings.sleeptime=udp_thread_sleeptime;
  udp_settings.motionThreads = udp_thread_dict.at("motion");
  udp_settings.sessionThreads = udp_thread_dict.at("session");
  udp_settings.lapDataThreads = udp_thread_dict.at("lap_data");
  udp_settings.eventThreads = udp_thread_dict.at("event");
  udp_settings.carsetupsThreads = udp_thread_dict.at("car_setups");
  udp_settings.participantsThreads = udp_thread_dict.at("participants");
  udp_settings.cartelemetryThreads = udp_thread_dict.at("telemetry");
  udp_settings.carstatusThreads = udp_thread_dict.at("car_status");
  std::shared_ptr<deepf1::MultiThreadedUDPHandler2018> udp_handler(new deepf1::MultiThreadedUDPHandler2018(udp_settings));
  // if(frame_handler)
  // {
  //   udp_handler->addPausedFunction(std::bind(&deepf1::MultiThreadedFrameGrabHandler::pause, frame_handler.get()));
  // }
  std::cout << "Created handlers" << std::endl;


  std::cout<<"Creating DataLogger" <<std::endl;
  std::shared_ptr<deepf1::F1DataLogger> dl( new deepf1::F1DataLogger( search_string, "127.0.0.1", udp_port) );
  std::cout<<"Created DataLogger" <<std::endl;
  


  std::string inp;
  std::cout<<"Enter anything to start capture" << std::endl;
  std::cin >> inp;
	std::cout << "Starting capture in " << initial_delay_time << " seconds." << std::endl;
  std::shared_ptr<deepf1::RebroadcastHandler2018> rbh;
  if(rebroadcast)
  {
    rbh.reset(new deepf1::RebroadcastHandler2018());
    dl->add2018UDPHandler(rbh);
  }
	std::this_thread::sleep_for(std::chrono::microseconds((long)std::round(initial_delay_time*1E6)));
  dl->add2018UDPHandler(udp_handler);
	dl->start(image_capture_frequency , frame_handler );
  
  std::cout<<"Started recording. Enter anything to stop"<<std::endl;
  std::string lol;
  std::cin>>lol;
	  //stop issuing new data to the handlers.
	dl->stop();
  //stop listening for data and just process whatever is left in the buffers.
  if(frame_handler)
	{
    frame_handler->stop();
  }
	udp_handler->stop();
	udp_handler->setSleepTime(10);
  //join with the main thread to keep the handlers in scope until all data has been written to file.
  if(frame_handler)
	{
	  frame_handler->join();
  }
	udp_handler->join(1);
  

}



