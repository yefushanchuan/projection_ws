#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/message_filter.h"
#include "tf2_ros/create_timer_ros.h"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "message_filters/subscriber.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

class TfListener : public rclcpp::Node
{
public:
  TfListener()
  : Node("tf_listener")
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    create_timer_ros_ = std::make_shared<tf2_ros::CreateTimerROS>(this->get_node_base_interface(),this->get_node_timers_interface());
    tf_buffer_->setCreateTimerInterface(create_timer_ros_);
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_,this);

    point_sub_.subscribe(this, "target_point");

    tf_filter_ = std::make_shared<tf2_ros::MessageFilter<geometry_msgs::msg::PointStamped>>(
      point_sub_, *tf_buffer_, "projection_frame", 10, this->get_node_logging_interface(),
      this->get_node_clock_interface());
    tf_filter_->registerCallback(
      &TfListener::pointCallback, this);
  }
private:
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  tf2_ros::CreateTimerROS::SharedPtr create_timer_ros_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  message_filters::Subscriber<geometry_msgs::msg::PointStamped> point_sub_;
  std::shared_ptr<tf2_ros::MessageFilter<geometry_msgs::msg::PointStamped>> tf_filter_;
  void pointCallback(const geometry_msgs::msg::PointStamped &msg)
  {
    geometry_msgs::msg::PointStamped point_out;
    try
    {
      point_out = tf_buffer_->transform(msg, "projection_frame");
    }
    catch (const tf2::TransformException & ex)
    {
      RCLCPP_INFO(this->get_logger(), "异常提示： %s", ex.what());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "参考系：%s，坐标值: (%.2f, %.2f, %.2f)",
                point_out.header.frame_id.c_str(),
                point_out.point.x,
                point_out.point.y,
                point_out.point.z);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TfListener>());
  rclcpp::shutdown();
  return 0;
}
