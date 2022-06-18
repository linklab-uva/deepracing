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
#include <tbb/concurrent_queue.h>
#include <tbb/task_group.h>
#include <atomic>
#include <yaml-cpp/yaml.h>
#include "f1_datalogger/proto/TimestampedPacketCarTelemetryData.pb.h"
#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"
#include <google/protobuf/util/json_util.h>
#include <iostream>
#include <fstream>

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
  ReplayDataset_2018DataGrabHandler(const fs::path& output_dir): output_dir_(output_dir),
    telemetry_counter(1)
  {
    car_index = 0;
  }
  inline bool isReady() override
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
    telemetry_data_queue_->push(data);
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
  void workerFunc()
  {
    google::protobuf::util::JsonOptions json_options;
    json_options.add_whitespace = true;
    json_options.always_print_primitive_fields = true;
    while(ready_ || !(telemetry_data_queue_->empty()) )    
    {
      deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData data;
      if(telemetry_data_queue_->try_pop(data))
      {
     //   std::cerr<<"Writing a packet to disk"<<std::endl;
        std::uint64_t counter = telemetry_counter.fetch_add(1, std::memory_order_relaxed);
        std::chrono::duration<double, std::milli> dt = (data.timestamp - begin);
        deepf1::twenty_eighteen::protobuf::TimestampedPacketCarTelemetryData data_pb;
        data_pb.set_timestamp(dt.count());
        data_pb.mutable_udp_packet()->CopyFrom(deepf1::twenty_eighteen::TwentyEighteenUDPStreamUtils::toProto(data.data));
        fs::path filename = output_dir_ / fs::path("packet_" + std::to_string(counter) + ".json");
        std::string json_string;
        google::protobuf::util::Status rc = google::protobuf::util::MessageToJsonString(data_pb, &json_string, json_options);
        std::ofstream ostream(filename.string(), std::fstream::out | std::fstream::trunc);
        ostream << json_string << std::endl;
        ostream.flush();
        ostream.close();
      }
    }

  }
  bool empty()
  {
    return telemetry_data_queue_->empty();
  }
  void run()
  {
    ready_ = true;
    thread_pool_->run(std::bind<void>(&ReplayDataset_2018DataGrabHandler::workerFunc,this));
  }
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override
  {
    std::cout<<"Initializing data handler"<<std::endl;
    thread_pool_.reset(new tbb::task_group);
    telemetry_data_queue_.reset( new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData> );
    ready_ = false;
	  car_index = 0;
    this->begin = begin;
  }
  std::chrono::high_resolution_clock::time_point begin;
  bool ready_;
private:
  uint8_t car_index;
  const fs::path output_dir_;
  std::shared_ptr< tbb::task_group > thread_pool_ ;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData> > telemetry_data_queue_;
  std::atomic<std::uint64_t> telemetry_counter;
};
int main(int argc, char** argv)
{
  std::string search, dataset_root_str;
  unsigned long dt;
  double delta;
  po::options_description desc("Allowed Options");

	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("output_dir,o", po::value<std::string>(&dataset_root_str)->required(), "Where to put recorded telemetry packets.")
			("timediff,t", po::value<unsigned long>(&dt)->required(), "How long to pause between updates (milliseconds).")
			("delta,d", po::value<double>(&delta)->required(), "How much to increase steering on each update")
			("search_string,s", po::value<std::string>(&search)->default_value("F1"), "How much to increase steering on each update")
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
  if ( fs::is_directory(dataset_root) )
  {
    std::cerr << dataset_root.string() << " is already a directory"<<std::endl;
    exit(0);
  }
  fs::create_directories(telemetry_dir);
  std::shared_ptr<deepf1::IF1FrameGrabHandler> image_handler;
  std::shared_ptr<ReplayDataset_2018DataGrabHandler> udp_handler( new ReplayDataset_2018DataGrabHandler(telemetry_dir) );
  std::shared_ptr<deepf1::F1Interface> controller = deepf1::F1InterfaceFactory::getDefaultInterface(1);
  deepf1::F1ControlCommand cc;
  cc.brake=0.0;
  cc.throttle=0.0;
  cc.steering=0.0;
  controller->setCommands(cc);
  
  deepf1::F1DataLogger dl(search);  
  std::chrono::duration sleeptime = std::chrono::milliseconds(dt);
  YAML::Node config;
  config["sleeptime"] = (double)dt;
  config["control_delta"] = delta;
  std::cout<<"Enter anything to start the test."<<std::endl;
  std::string asdf;
  std::cin >> asdf;
  std::cout<<"Starting test in 3 seconds."<<std::endl;
  dl.add2018UDPHandler(udp_handler);
  dl.start(35.0, image_handler);
  udp_handler->run();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::chrono::duration<double, std::milli> starttime = (std::chrono::high_resolution_clock::now() - udp_handler->begin);
  while(cc.steering<=1.0)
  {
    cc.steering+=delta;
    controller->setCommands(cc);
    std::this_thread::sleep_for(sleeptime);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  udp_handler->ready_=false;
  std::cout<<"Cleaning up remaining packets"<<std::endl;
  config["time_start"] = starttime.count();
  std::ofstream configout((dataset_root/fs::path("config.yaml")).string(), std::fstream::out | std::fstream::trunc);
  configout << config;
  configout.flush();
  configout.close();
  while(!udp_handler->empty())
  {

  }


 

}

