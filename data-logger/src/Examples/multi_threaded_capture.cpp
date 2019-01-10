/*
 * multi_threaded_capture.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */
#include "f1_datalogger.h"
#include "image_logging/common/multi_threaded_framegrab_handler.h"
#include "udp_logging/common/multi_threaded_udp_handler.h"
#include <boost/program_options.hpp>
//#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace scl = SL::Screen_Capture;
namespace po = boost::program_options;
void exit_with_help(po::options_description& desc)
{
        std::stringstream ss;
        ss << "F1 Simulated Telemetry Server. Command line arguments are as follows:" << std::endl;
        desc.print(ss);
        std::printf("%s", ss.str().c_str());
        exit(0); // @suppress("Invalid arguments")
}
int main(int argc, char** argv)
{
	using namespace deepf1;
  std::string search, images_folder, udp_folder;
  unsigned int image_threads, udp_threads;
  // if (argc > 1)
  // {
  //   search = std::string(argv[1]);
  // }
  // double capture_frequency = 10.0;
  // if (argc > 2)
  // {
  //   capture_frequency = atof(argv[2]);
  // }
  po::options_description desc("Allowed Options");
  try{
    desc.add_options()
      ("help,h", "Displays options and exits")
      ("search_string,s", po::value<std::string>(&search)->required(), "Search string to find application")
      ("images_folder,i", po::value<std::string>(&images_folder)->default_value("images"), "Folder to log images to")
      ("udp_folder,u", po::value<std::string>(&udp_folder)->default_value("udp_data"), "Folder to log udp data to")
      ("image_threads", po::value<unsigned int>(&image_threads)->default_value(2), "Folder to log images to")
      ("udp_threads", po::value<unsigned int>(&udp_threads)->default_value(2), "Folder to log udp data to")
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
  std::cout<<"Creating handlers" <<std::endl;
  std::shared_ptr<deepf1::MultiThreadedFrameGrabHandler> frame_handler(new deepf1::MultiThreadedFrameGrabHandler(images_folder, image_threads));
  std::shared_ptr<deepf1::MultiThreadedUDPHandler> udp_handler(new deepf1::MultiThreadedUDPHandler(udp_folder, udp_threads));
  std::cout<<"Creating DataLogger" <<std::endl;
  deepf1::F1DataLogger dl(search, frame_handler, udp_handler);
  std::cout<<"Created DataLogger" <<std::endl;
  std::string inp;
  std::cout<<"Enter any key to start " << std::endl;
  std::cin >> inp;
  dl.start(25.0);

  std::cout<<"Enter any key to end " << std::endl;
  std::cin >> inp;
}



