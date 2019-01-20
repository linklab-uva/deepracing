/*
 * protobuf_test.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */

#include "F1UDPData.pb.h"
#include <fstream>

int main(int argc, char** argv)
{
  deepf1::protobuf::F1UDPData data;
  data.set_throttle(1.0);
  data.set_brake(0.0);
  data.set_steering(-1.0);

  std::ofstream stream("file.pb");
  data.SerializeToOstream(&stream);
  stream.close();


  deepf1::protobuf::F1UDPData data_in;

  std::ifstream stream_in("file.pb");
  data_in.ParsePartialFromIstream(&stream_in);
  stream_in.close();
  printf("Read some data. Brake: %f. Throttle: %f. Steering: %f\n", data_in.brake(), data_in.throttle(), data_in.steering());

}
