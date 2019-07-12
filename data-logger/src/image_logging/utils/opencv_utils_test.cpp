#include <iostream>
#include "f1_datalogger/image_logging/utils/opencv_utils.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <memory>
#include <boost/array.hpp>
#include <boost/asio.hpp>
using boost::asio::ip::tcp;
int main(int argc, char** argv)
{
  std::string imfile(argv[1]);
  double imresizefactor = std::stod(std::string(argv[2]));
  std::string server(argv[3]);
  int port = std::stoi(std::string(argv[4]));
  cv::Mat im = cv::imread(imfile, cv::IMREAD_UNCHANGED);
  cv::resize(im, im, cv::Size((int)std::round(imresizefactor * im.cols), (int)std::round(imresizefactor * im.rows)));

  cv::namedWindow("input", cv::WINDOW_AUTOSIZE);
  cv::imshow("input", im);
  cv::waitKey(0);
  cv::destroyAllWindows();
  deepf1::protobuf::images::Image im_proto = deepf1::OpenCVUtils::cvimageToProto(im);
  std::printf("Image has %u rows and %u columns.\n", uint32_t(im_proto.rows()), uint32_t(im_proto.cols()));
  std::unique_ptr<std::string> jsonstring(new std::string);
  google::protobuf::util::JsonOptions opshinz;
  opshinz.always_print_primitive_fields = true;
  opshinz.add_whitespace = true;
  google::protobuf::util::Status rc = google::protobuf::util::MessageToJsonString(im_proto, jsonstring.get(), opshinz);
  std::ofstream fout("imageout.json");
  fout << (*jsonstring);
  fout.close();
  //jsonstring.reset(new std::string("Hello World!"));
  boost::asio::io_service  io_service;
  tcp::socket socket(io_service);
  socket.connect(boost::asio::ip::tcp::endpoint(boost::asio::ip::address_v4::from_string(server), port));
  bool success = false;
  while (!success)
  {
   // uint32_t numbytes = jsonstring->length();
    uint32_t numbytes = im_proto.ByteSize();
    std::unique_ptr<char[]> buffer(new char[sizeof(uint32_t) + (uint64_t)numbytes]);
    memcpy(buffer.get(), &numbytes, sizeof(uint32_t));
    im_proto.SerializeToArray(&((buffer.get())[sizeof(uint32_t)]), numbytes);

    boost::system::error_code error;
    size_t len = boost::asio::write(socket,boost::asio::buffer(buffer.get(), (uint64_t)numbytes + (uint64_t)4), error);
    std::cout << "Send result message: " << error.message() << std::endl;

    uint32_t recv_size;
    size_t lenrecv =  boost::asio::read(socket, boost::asio::buffer(&recv_size, sizeof(uint32_t)));
    std::cout << "Receive size result message: " << error.message() << std::endl;
    std::unique_ptr<uchar[]> reply_buffer(new uchar[(size_t)recv_size]);
    deepf1::protobuf::images::Image imProto;
    lenrecv = boost::asio::read(socket, boost::asio::buffer(reply_buffer.get(), recv_size));
    
    success = imProto.ParseFromArray(reply_buffer.get(), recv_size);
    cv::Mat server_image = deepf1::OpenCVUtils::protoImageToCV(imProto);
    cv::namedWindow("converted_output", cv::WINDOW_AUTOSIZE);
    cv::imshow("converted_output", server_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  socket.close();
  return 0;
}