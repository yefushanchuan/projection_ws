#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"

class StaticTfBroadcaster : public rclcpp::Node
{
public:
  StaticTfBroadcaster(char ** argv)
  : Node("static_tf_broadcaster")
  {
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    send_static_transform(argv);
  }
private:
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  void send_static_transform(char ** argv)
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = argv[7];
    t.child_frame_id = argv[8];
    t.transform.translation.x = atof(argv[1]);
    t.transform.translation.y = atof(argv[2]);
    t.transform.translation.z = atof(argv[3]);
    tf2::Quaternion q;
    q.setRPY(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();
    static_broadcaster_->sendTransform(t);
    RCLCPP_INFO(this->get_logger(), "发送静态坐标变换：父坐标系：%s, 子坐标系：%s", argv[7], argv[8]);
  }
};

int main(int argc, char ** argv)
{
  if(argc != 9)
  {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "传入参数不合法！");
    return 1;
  }
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StaticTfBroadcaster>(argv));
  rclcpp::shutdown();
  return 0;
}
