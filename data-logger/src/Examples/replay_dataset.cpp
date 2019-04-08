/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/f1_datalogger.h"
 //#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vJoy++/vjoy.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <f1_datalogger/udp_logging/utils/udp_stream_utils.h>
#include <f1_datalogger/post_processing/post_processing_utils.h>
#include <google/protobuf/util/json_util.h>
#include <boost/thread.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tbb/concurrent_queue.h>
#include <tbb/task_group.h>
namespace po = boost::program_options;
namespace fs = boost::filesystem;
void exit_with_help(po::options_description& desc)
{
	std::stringstream ss;
	ss << "F1 Replay Dataset. Command line arguments are as follows:" << std::endl;
	desc.print(ss);
	std::printf("%s", ss.str().c_str());
	exit(0);
}
void countdown(unsigned int seconds, std::string text = "")
{
	std::cout << text << std::endl;
	for (unsigned int i = seconds; i > 0; i--)
	{
		std::cout << i << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}
bool sortByTimestamp(const deepf1::protobuf::TimestampedUDPData& a, const deepf1::protobuf::TimestampedUDPData& b)
{
	return a.udp_packet().m_time() < b.udp_packet().m_time();
}
class ReplayDataset_DataGrabHandler : public deepf1::IF1DatagrabHandler
{
public:
	ReplayDataset_DataGrabHandler(boost::barrier& bar, unsigned int num_threads) :
		bar_(bar), waiting_(true), counter_(1), num_threads_(num_threads)
	{
	}
	virtual ~ReplayDataset_DataGrabHandler()
	{
		stop();
	}
	bool isReady() override
	{
		return true;
	}
	void stop()
	{
		running_ = false;
	}
	void workerFunc()
	{
		while (running_ || !queue_->empty())
		{
			if (queue_->empty())
			{
				continue;
			}
			deepf1::TimestampedUDPData data;
			//std::lock_guard<std::mutex> lk(queue_mutex_);
			if (!queue_->try_pop(data))
			{
				continue;
			}
			deepf1::protobuf::TimestampedUDPData data_pb;
			data_pb.mutable_udp_packet()->CopyFrom(deepf1::UDPStreamUtils::toProto(data.data));
			data_pb.set_timestamp((google::protobuf::uint64)(std::chrono::duration_cast<std::chrono::milliseconds>(data.timestamp - begin).count()));
			std::string json;
			google::protobuf::util::JsonOptions opshinz;
			opshinz.always_print_primitive_fields = true;
			opshinz.add_whitespace = true;
			google::protobuf::util::MessageToJsonString(data_pb, &json, opshinz);
			std::string json_file = "packet_" + std::to_string(counter_.fetch_and_increment()) + ".pb.json";
			std::string json_fn = (dir / fs::path(json_file)).string();
			ostream->open(json_fn.c_str(), std::ofstream::out);
			ostream->write(json.c_str(), json.length());
			ostream->flush();
			ostream->close();
		}
	}
	void handleData(const deepf1::TimestampedUDPData& data) override
	{
		if (waiting_ && data.data.m_lapTime<=1E-3)
		{
			bar_.wait();
			waiting_ = false;
		}
		else if (!waiting_)
		{
			queue_->push(data);
		}
	}
	void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override
	{
		idx = 0;
		dir = fs::path("playback");
		ostream.reset(new std::ofstream);
		if (!fs::is_directory(dir))
		{
			fs::create_directory(dir);
		}
		this->begin = begin;
		running_ = true;
		queue_.reset(new tbb::concurrent_queue<deepf1::TimestampedUDPData>);
		thread_pool_.reset(new tbb::task_group);
		for (unsigned int i = 1; i <= num_threads_; i++)
		{
			thread_pool_->run(std::bind<void>(&ReplayDataset_DataGrabHandler::workerFunc, this));
		}
	}
	std::chrono::high_resolution_clock::time_point getBegin()
	{
		return begin;
	}
private:
	std::shared_ptr< tbb::concurrent_queue< deepf1::TimestampedUDPData> > queue_;
	std::shared_ptr< tbb::task_group> thread_pool_;
	bool running_;
	std::unique_ptr<std::ofstream> ostream;
	fs::path dir;
	std::chrono::high_resolution_clock::time_point begin;
	boost::barrier& bar_;
	bool waiting_;
	unsigned long idx;
	unsigned int num_threads_;
	tbb::atomic<unsigned long> counter_;
};
class ReplayDataset_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
	ReplayDataset_FrameGrabHandler(double freq=60.0) 
		: captureFreq(freq)
	{
	}
	virtual ~ReplayDataset_FrameGrabHandler()
	{
		
	}
	bool isReady() override
	{
		return true;
	}
	void handleData(const deepf1::TimestampedImageData& data) override
	{

		cv::Mat img_cv_video;
		cv::cvtColor(data.image, img_cv_video, cv::COLOR_BGRA2BGR);
		//cv::imshow(window_name, img_cv_video);
		//cv::Size s = img_cv_video.size();
	   // std::cout<<"Image is: " << s.height<< " X " << s.width << std::endl;
		video_writer_->write(img_cv_video);
	}
	void init(const std::chrono::high_resolution_clock::time_point& begin, const cv::Size& window_size) override
	{
		video_writer_.reset(new cv::VideoWriter("playback.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), captureFreq, window_size));
		this->begin = begin;
	}
	std::chrono::high_resolution_clock::time_point getBegin()
	{
		return begin;
	}
	const double captureFreq;
private:
	std::chrono::high_resolution_clock::time_point begin;
	std::shared_ptr<cv::VideoWriter> video_writer_;
};
void setControls(const std::vector<double>& laptimes, const std::vector<double>& steering,
	const std::vector<double>& throttle, const std::vector<double>& brake, 
	const double& time, const double& max_vjoy, vjoy_plusplus::JoystickPosition& js)
{
	
}
int main(int argc, char** argv)
{
	std::unique_ptr<std::string> search(new std::string);
	std::unique_ptr<std::string> dir(new std::string);
	unsigned int num_threads;
	po::options_description desc("Allowed Options");

	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("search_string,s", po::value<std::string>(search.get())->default_value("2017"), "Search string to find the window name for F1 2017")
			("data_dir,d", po::value<std::string>(dir.get())->required(), "Directory to look for stored UDP data")
			("num_threads,t", po::value<unsigned int>(&num_threads)->default_value(1), "Number of threads to spawn for recording the resulting game data.")
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
	vjoy_plusplus::vJoy vjoy(1);
	vjoy_plusplus::JoystickPosition js;
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (unsigned int)std::round(0.5*(double)(min + max));
	js.lButtons = 0x00000000;
	js.wAxisX = 0;
	js.wAxisY = 0;
	js.wAxisXRot = 0;
	js.wAxisYRot = 0;
	vjoy.update(js);
	std::vector<deepf1::protobuf::TimestampedUDPData> data = deepf1::post_processing::PostProcessingUtils::parseUDPDirectory(*dir);
	std::vector<deepf1::protobuf::TimestampedUDPData> sorted_data;
	std::sort(data.begin(), data.end(), sortByTimestamp);
	std::cout << "Got a dataset with " << data.size() << " elements." << std::endl;
	for (unsigned int i = 0; i < data.size(); i++)
	{
		deepf1::protobuf::TimestampedUDPData datapoint = data.at(i);
		std::printf("Tag %u has Steering: %f, Throttle: %f, Brake: %f. Lap Time: %f\n", i, datapoint.udp_packet().m_steer(), datapoint.udp_packet().m_throttle(), datapoint.udp_packet().m_brake(), datapoint.udp_packet().m_laptime());
		if (datapoint.udp_packet().m_laptime() < 1E-4)
		{
			std::printf("Found the starting packet at %u with Steering: %f, Throttle: %f, Brake: %f. Lap Time: %f\n", i, datapoint.udp_packet().m_steer(), datapoint.udp_packet().m_throttle(), datapoint.udp_packet().m_brake(), datapoint.udp_packet().m_laptime());
			std::vector<deepf1::protobuf::TimestampedUDPData>::const_iterator first = data.begin() + i;
			std::vector<deepf1::protobuf::TimestampedUDPData>::const_iterator last = data.end();
			sorted_data = std::vector<deepf1::protobuf::TimestampedUDPData>(first, last);
			break;
		}
	}
	std::cout << "Extracted with " << sorted_data.size() << " elements from dataset." << std::endl;
	std::vector<double> laptimes, steering, throttle, brake;
	double timesteps = 2.0;
	for (unsigned int i = 1; i < sorted_data.size(); i++)
	{
		double currentTime = sorted_data.at(i - 1).udp_packet().m_laptime();
		double currentSteer = sorted_data.at(i - 1).udp_packet().m_steer();
		double currentThrottle = sorted_data.at(i - 1).udp_packet().m_throttle();
		double currentBrake = sorted_data.at(i - 1).udp_packet().m_brake();

		double timeUB = sorted_data.at(i).udp_packet().m_laptime();
		double steerUB = sorted_data.at(i).udp_packet().m_steer();
		double throttleUB = sorted_data.at(i).udp_packet().m_throttle();
		double brakeUB = sorted_data.at(i).udp_packet().m_brake();

		double DT;
		double throttle2;
		double throttle1;
		double brake2;
		double brake1;
		double steer_slope;
		double throttle_slope;
		double brake_slope;
		double steer_command;
		double throttle_command;
		double brake_command;
		laptimes.push_back(currentTime);
		steering.push_back(currentSteer);
		throttle.push_back(currentThrottle);
		brake.push_back(currentBrake);

		DT = timeUB - currentTime;
		steer_slope = (steerUB - currentSteer) / DT;
		throttle_slope = (throttleUB - currentThrottle) / DT;
		brake_slope = (brakeUB - currentBrake) / DT;
		double dt = DT / timesteps;
		for (double time = currentTime + dt; time < timeUB; time += dt)
		{
			double dt = time - currentTime;
			steer_command = currentSteer + steer_slope * dt;
		    throttle_command = currentThrottle + throttle_slope * dt;
			brake_command = currentBrake + brake_slope * dt;
			laptimes.push_back(time);
			steering.push_back(steer_command);
			throttle.push_back(throttle_command);
			brake.push_back(brake_command);
		}
	}
	double max_vjoysteer = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoythrottle = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoybrake = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	double middle_vjoysteer = max_vjoysteer / 2.0;
	std::chrono::high_resolution_clock clock;
	boost::barrier bar(2);
	std::shared_ptr<ReplayDataset_FrameGrabHandler> image_handler(new ReplayDataset_FrameGrabHandler());
	std::shared_ptr<ReplayDataset_DataGrabHandler> udp_handler(new ReplayDataset_DataGrabHandler(boost::ref(bar), num_threads));
	std::unique_ptr<deepf1::F1DataLogger> dl(new deepf1::F1DataLogger(*search, image_handler, udp_handler));
	dl->start(image_handler->captureFreq);
	double time;
    int idx;
	time = 0.0;
	double maxtime = laptimes.back();
	std::chrono::high_resolution_clock::time_point begin;
	bar.wait();
	begin = clock.now();
	//dl.reset();
	//Best fit line is : y = -16383.813867*x + 16383.630437
	float fake_zero=1E-3;
	float positive_deadband = fake_zero, negative_deadband = -fake_zero;
	while (time< maxtime)
	{
		time = 1E-6*((double)(std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - begin).count()));
		idx = (std::upper_bound(laptimes.begin(), laptimes.end(), time) - laptimes.begin());
		//if (idx >= laptimes.size() - (unsigned int)timesteps - 1)
		//{
		//	break;
		//}
		//vjoy = 32767.253637*f1 + 0.128575

		if (steering[idx] > positive_deadband)
		{
			js.wAxisX = (unsigned int)std::round(max_vjoysteer*steering[idx]);
			js.wAxisY = 0;
		}
		else if (steering[idx] < negative_deadband)
		{
			js.wAxisX = 0;
			js.wAxisY = (unsigned int)std::round(-max_vjoysteer*steering[idx]);
		}
		else
		{
			js.wAxisX = 0;
			js.wAxisY = 0;
		}
		js.wAxisXRot = (unsigned int)std::round(max_vjoythrottle*throttle[idx]);
		js.wAxisYRot = (unsigned int)std::round(max_vjoybrake*brake[idx]);
		vjoy.update(js);
	}
	std::cout << "Thanks for Playing! Enter anything to exit." << std::endl;
	js.wAxisX = 0;
	js.wAxisY = 0;
	js.wAxisXRot = 0;
	js.wAxisYRot = 0;
	vjoy.update(js);
	udp_handler->stop();
	std::string s;
	std::cin >> s;
	//cv::waitKey(0);

}

