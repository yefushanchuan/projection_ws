#include <chrono>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "object3d_msgs/msg/object3_d_array.hpp"

using namespace std::chrono_literals;

class ImageViewerTalker : public rclcpp::Node
{
public:
  ImageViewerTalker()
  : Node("image_viewer_talker"), count_(0)
  {
    this->declare_parameter<double>("fx", 1612.8);
    this->declare_parameter<double>("fy", 1612.8);
    this->declare_parameter<double>("cx", 640.0); 
    this->declare_parameter<double>("cy", 360.0);
    this->declare_parameter<int>("point_radius", 15);

    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_topic", 10);

    // 创建自定义消息订阅器，监听 Circle 消息
    points_sub_  = this->create_subscription<object3d_msgs::msg::Object3DArray>(
      "target_points_projection_array", 10,
      std::bind(&ImageViewerTalker::points_callback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "Image viewer talker started.");
  }

private:
  void points_callback(const object3d_msgs::msg::Object3DArray::SharedPtr msg)
  {
    double fx = this->get_parameter("fx").as_double();
    double fy = this->get_parameter("fy").as_double();
    double cx = this->get_parameter("cx").as_double();
    double cy = this->get_parameter("cy").as_double();
    int radius = this->get_parameter("point_radius").as_int();

    // 黑图初始化
    cv::Mat image = cv::Mat::zeros(1080, 1920, CV_8UC3);

    for (const auto & obj : msg->objects)
    {
      double X = obj.point.x;
      double Y = obj.point.y;
      double Z = obj.point.z;

      // 简单的深度保护：如果 Z 太小（比如0），强制设为 1.0，防止除以0崩溃
      // 如果你的场景是纯 2D 平面移动，Z 可能一直是 0 或 1
      if (std::abs(Z) < 0.001) Z = 1.0; 

      // --- 核心投影公式 (3D -> 2D) ---
      // 这里的逻辑是：物体越远(Z越大)，坐标越靠近中心；fx/fy 控制放缩倍数
      // 如果你想去掉透视效果(不管Z多远，大小位置只看XY)，就把下面的 /Z 去掉
      int u = static_cast<int>((fx * X / Z) + cx); 
      int v = static_cast<int>((fy * Y / Z) + cy);

      // 绘制点
      // 只要计算出的点在屏幕范围内，就画出来
      if (u >= 0 && u < 1920 && v >= 0 && v < 1080) {
        // 参数说明: 图像, 中心坐标, 半径(固定), 颜色(白色), 填充(-1)
        cv::circle(image, cv::Point(u, v), radius, cv::Scalar(255, 255, 255), -1);
      }
    }

    std_msgs::msg::Header header;
    header.stamp = this->now();
    header.frame_id = "projector_frame";

    try {
      auto ros_image = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
      image_pub_->publish(*ros_image);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<object3d_msgs::msg::Object3DArray>::SharedPtr points_sub_;
  int count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageViewerTalker>());
  rclcpp::shutdown();
  return 0;
}
