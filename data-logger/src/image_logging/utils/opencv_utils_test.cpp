#include <iostream>
#include "f1_datalogger/image_logging/utils/opencv_utils.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
int main(int argc, char** argv)
{
  std::string imfile(argv[1]);
  double imresizefactor = std::stod(std::string(argv[2]));
  cv::Mat im = cv::imread(imfile, cv::IMREAD_UNCHANGED);
  cv::resize(im, im, cv::Size(), imresizefactor, imresizefactor);
  return 0;
}