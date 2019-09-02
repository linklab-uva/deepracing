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
#include <filesystem>
namespace fs = std::filesystem;
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
  std::string search_string, image_folder, image_extension, udp_folder, config_file, root_directory, driver_name;
  unsigned int image_threads, udp_port;
  float image_capture_frequency, initial_delay_time;
  bool spectating, use_json, init_paused;
  double capture_region_ratio;


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
  image_capture_frequency = config_node["image_capture_frequency"].as<float>();
  initial_delay_time = config_node["initial_delay_time"].as<float>();
  spectating = config_node["spectating"].as<bool>(false);
  init_paused = config_node["init_paused"].as<bool>(false);
  use_json = config_node["use_json"].as<bool>(true);
  capture_region_ratio = config_node["capture_region_ratio"].as<double>(1.0);
  
  config_node["search_string"] = search_string;
  config_node["images_folder"] = image_folder;
  config_node["udp_folder"] = udp_folder;
  config_node["image_extension"] = image_extension;
  config_node["image_threads"] = image_threads;
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
  yamlout.open((root_dir/fs::path("dataset_config.yaml")).string(), std::fstream::out | std::fstream::trunc);
  yamlout<<config_node;
  yamlout.close();
  


  std::cout<<"Creating handlers" <<std::endl;
  deepf1::MultiThreadedFrameGrabHandlerSettings settings;
  settings.image_extension=image_extension;
  settings.images_folder=(root_dir/fs::path(image_folder)).string();
  settings.thread_count=image_threads;
  settings.write_json=use_json;
  settings.capture_region_ratio=capture_region_ratio;
  std::shared_ptr<deepf1::MultiThreadedFrameGrabHandler> frame_handler(new deepf1::MultiThreadedFrameGrabHandler(settings));
  if (init_paused)
  {
    std::cout<<"Initially pausing the frame-grab loop"<<std::endl;
    frame_handler->pause();
  }
  deepf1::MultiThreadedUDPHandler2018Settings udp_settings;
  udp_settings.write_json=use_json;
  udp_settings.udp_directory=(root_dir/fs::path(udp_folder)).string();
  std::shared_ptr<deepf1::MultiThreadedUDPHandler2018> udp_handler(new deepf1::MultiThreadedUDPHandler2018(udp_settings));
  udp_handler->addPausedFunction(std::bind(&deepf1::MultiThreadedFrameGrabHandler::pause, frame_handler.get()));
  std::cout << "Created handlers" << std::endl;


  std::cout<<"Creating DataLogger" <<std::endl;
  std::shared_ptr<deepf1::F1DataLogger> dl( new deepf1::F1DataLogger( search_string, "127.0.0.1", udp_port) );
  std::cout<<"Created DataLogger" <<std::endl;


  std::string inp;
  std::cout<<"Enter anything to start capture" << std::endl;
  std::cin >> inp;
	std::cout << "Starting capture in " << initial_delay_time << " seconds." << std::endl;
	std::this_thread::sleep_for(std::chrono::microseconds((long)std::round(initial_delay_time*1E6)));
	dl->start(image_capture_frequency, udp_handler  , frame_handler );
  unsigned int ycount = 0;
  unsigned int bcount = 0;
  std::cout << "Recording. Push Y to pause. Push left thumbstick to unpause. Push right thumbstick to unpause to Exit." << std::endl;
  DirectX::GamePad gp;
  DirectX::GamePad::State gpstate;
  std::function<bool()> isUnpausePressed = std::bind(&DirectX::GamePad::State::IsLeftStickPressed, &gpstate);
  std::function<bool()> pause = [&gpstate, &spectating]() {return (gpstate.IsStartPressed() || (spectating && (gpstate.IsYPressed() || gpstate.IsStartPressed() || gpstate.IsRightTriggerPressed() || gpstate.IsLeftTriggerPressed()
    || gpstate.IsRightShoulderPressed() || gpstate.IsLeftShoulderPressed() || gpstate.IsBPressed() || gpstate.IsXPressed()) ) ); };
  while (true)
  {
    gpstate = gp.GetState(0);
    if (pause())
    {
      printf("Pausing %u\n", ++ycount);
      frame_handler->pause();
    }
    if (gpstate.IsStartPressed())
    {
      printf("Start is pressed. Pausing %u\n", ++ycount);
      frame_handler->pause();
    }
    if (isUnpausePressed())
    {
      printf("Unpausing %u\n", ++bcount);
      frame_handler->resume();
    }
    if (gpstate.IsRightStickPressed())
    {
      printf("%s","Right stick is pressed. Exiting\n");
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
	  //stop issuing new data to the handlers.
	  dl->stop();
  //stop listening for data and just process whatever is left in the buffers.
	frame_handler->stop();
	udp_handler->stop();
  //join with the main thread to keep the handlers in scope until all data has been written to file.
	frame_handler->join();
	udp_handler->join(1);
  

}



