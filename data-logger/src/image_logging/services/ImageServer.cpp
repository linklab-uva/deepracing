#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "f1_datalogger/proto/DeepF1_RPC.grpc.pb.h"
#include "f1_datalogger/image_logging/utils/opencv_utils.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
namespace fs = std::filesystem;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using deepf1::protobuf::images::ImageRequest;
using deepf1::protobuf::images::ImageService;
using deepf1::protobuf::images::Image;

class ImageServerImpl final : public ImageService::Service 
{
//private:
public:
  void setImageFolder(const std::string& image_dir)
  {
    folder_ = fs::path(image_dir);
  }
  void setResizeFactor(const double& resize_factor)
  {
    resize_factor_ = resize_factor;
  }
private:
  Status GetImage(ServerContext* context, const ImageRequest* request,
    Image* reply) override
  {

    std::cerr << "Processing request for image key: " + request->key() << std::endl;
    fs::path fullpath = folder_ / fs::path(request->key());
    cv::Mat im;
    try {
      cv::resize(cv::imread(fullpath.string(), cv::IMREAD_UNCHANGED), im, cv::Size(), resize_factor_, resize_factor_, cv::INTER_AREA);
      cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
      if (im.empty())
      {
        Status status(grpc::StatusCode::INVALID_ARGUMENT, "Image " + request->key() + " doesn't exist.");
        return status;
      }
    }
    catch (const cv::Exception& ex)
    {
      Status status(grpc::StatusCode::INVALID_ARGUMENT, "Image " + request->key() + " doesn't exist.");
      return status;
    }

    deepf1::protobuf::images::Image imageres = deepf1::OpenCVUtils::cvimageToProto(im);
    //imageres.set_rows(10);
   // imageres.set_cols(25);
    //imageres.set_image_data("This is not really an image.");
    reply->CopyFrom(imageres);
    return Status::OK;
  }
  double resize_factor_;
  fs::path folder_;

};


int main(int argc, char** argv)
{
  if (argc < 5)
  {
    std::cout << "Usage: \"f1_datalogger_image_server <address> <port number> <image_directory> <resize_factor>\"" << std::endl;
    exit(0);
  }
  std::string server_address(std::string(argv[1]) + ":" + std::string(argv[2]));
  ImageServerImpl service;
  service.setImageFolder(std::string(argv[3]));
  service.setResizeFactor(std::stof(std::string(argv[4])));

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
  return 0;
}