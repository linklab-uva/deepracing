/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */
#include "f1_datalogger/f1_datalogger.h"
//#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <Eigen/Geometry>
#include "f1_datalogger/udp_logging/utils/eigen_utils.h"
#include "f1_datalogger/controllers/f1_interface_factory.h"
#include <boost/program_options.hpp>
#include <filesystem>

namespace scl = SL::Screen_Capture;
namespace po = boost::program_options;
namespace fs = std::filesystem;
void exit_with_help(po::options_description& desc)
{
	std::stringstream ss;
	ss << "F1 Datalogger Replay Dataset. Command line arguments are as follows:" << std::endl;
	desc.print(ss);
	std::printf("%s", ss.str().c_str());
	exit(0); 
}
class ReplayDataset_2018DataGrabHandler : public deepf1::IF12018DataGrabHandler
{
public:
  ReplayDataset_2018DataGrabHandler()
  {
    car_index = 0;
  }
  bool isReady() override
  {
    return ready_;
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data) override
  {
    // const int8_t& steer_value = data.data.m_carTelemetryData[car_index].m_steer;
	// std::printf("Got a telemetry packet. Steering Value: %d\n", steer_value);
	// std::printf("Got a telemetry packet. Steering Ratio: %f\n", ((double)steer_value)/100.0);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data) override
  {
    
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) override
  {
    
  }
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override
  {
    ready_ = true;
	car_index = 0;
    this->begin = begin;
  }
private:
  bool ready_;
  std::chrono::high_resolution_clock::time_point begin;
  uint8_t car_index;
};
int main(int argc, char** argv)
{
  std::string search, dataset_root_str;
  po::options_description desc("Allowed Options");

	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("dataset_root,r", po::value<std::string>(&dataset_root_str)->required(), "Root directory of dataset to replay.")
			("search_string,s", po::value<std::string>(&search)->default_value("F1"), "Root directory of dataset to replay.")
			;
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
		if (vm.find("help") != vm.end()) 
		{
			exit_with_help(desc);
		}
	}
	catch (const boost::exception& e) {
		exit_with_help(desc);
	}
  fs::path dataset_root(dataset_root_str);
  fs::path udp_root = dataset_root / fs::path("udp_data");
  fs::path telemetry_dir = udp_root / fs::path("car_telemetry_packets");
  if (! fs::is_directory(telemetry_dir))
  {
    std::cerr << dataset_root.string() << " is not a valid F1 dataset"<<std::endl;
    exit(0);
  }

  fs::path replayed_udp_dir = dataset_root / fs::path("replayed_udp_data");
  fs::path replayed_telemetry_dir = replayed_udp_dir / fs::path("car_telemetry_packets");
  fs::create_directories(replayed_telemetry_dir);
  std::shared_ptr<deepf1::IF1FrameGrabHandler> image_handler;
  std::shared_ptr<ReplayDataset_2018DataGrabHandler> udp_handler( new ReplayDataset_2018DataGrabHandler() );
  std::shared_ptr<deepf1::F1Interface> controller = deepf1::F1InterfaceFactory::getDefaultInterface(1);
  
  deepf1::F1DataLogger dl(search);  
  dl.start(35.0, udp_handler, image_handler);
  std::cout<<"Enter anything to exit."<<std::endl;
  std::string asdf;
  std::cin >> asdf;
 

}

