#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/point_stamped.hpp"

class PointTfBroadcaster : public rclcpp::Node
{
public:
  PointTfBroadcaster()
  : Node("point_tf_broadcaster")
  {
    point_broadcaster_ = this->create_publisher<geometry_msgs::msg::PointStamped>("point", 10);
    point_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/target_depth", 10,
      std::bind(&PointTfBroadcaster::point_callback, this, std::placeholders::_1));
  }
private:
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr point_broadcaster_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr point_sub_;
    void point_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    geometry_msgs::msg::PointStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = msg->header.frame_id;
    t.point = msg->point;

    point_broadcaster_->publish(t);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointTfBroadcaster>());
  rclcpp::shutdown();
  return 0;
}
