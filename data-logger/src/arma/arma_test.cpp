#include <f1_datalogger/udp_logging/utils/eigen_utils.h>
#include <iostream>
int main(int argc, char** argv)
{
  Eigen::MatrixXd A = 
    deepf1::EigenUtils::loadArmaTxt(std::string(argv[1]));
  std::cout << A << std::endl;

}