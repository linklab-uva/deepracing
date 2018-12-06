/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger.h"

class OpenCV_Viewer_Example_Handler : public deepf1::IF1FrameGrabHandler
{
public:
  OpenCV_Viewer_Example_Handler()
  {

  }
  virtual ~OpenCV_Viewer_Example_Handler()
  {

  }
  bool isReady() override
  {
    return true;
  }
  void handleData(const deepf1::TimestampedImageData& data) override
  {

  }
};

int main(int argc, char** argv)
{
  std::string search;
  if (argc > 1)
  {
    search = std::string(argv[1]);
  }
  else
  {
    search = "CMake";
  }
  std::shared_ptr<OpenCV_Viewer_Example_Handler> handler(new OpenCV_Viewer_Example_Handler());
  deepf1::F1DataLogger dl(search, handler);
}

