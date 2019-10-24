#include "rclcpp/rclcpp.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("ObiWan");

  RCLCPP_INFO(node->get_logger(),
              "Help me Obi-Wan Kenobi, you're my only hope");

  rclcpp::shutdown();
  return 0;
}