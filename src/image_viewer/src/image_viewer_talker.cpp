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
    this->declare_parameter<double>("min_z_threshold", 0.5);
    this->declare_parameter<int>("image_width", 1280);
    this->declare_parameter<int>("image_height", 720);

    fx_ = this->get_parameter("fx").as_double();
    fy_ = this->get_parameter("fy").as_double();
    cx_ = this->get_parameter("cx").as_double();
    cy_ = this->get_parameter("cy").as_double();
    radius_ = this->get_parameter("point_radius").as_int();
    min_z_ = this->get_parameter("min_z_threshold").as_double();
    image_width_ = this->get_parameter("image_width").as_int();
    image_height_ = this->get_parameter("image_height").as_int();

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
    // 黑图初始化
    cv::Mat image = cv::Mat::zeros(image_height_, image_width_, CV_8UC3);

    for (const auto & obj : msg->objects)
    {
      double X = obj.point.x;
      double Y = obj.point.y;
      double Z = obj.point.z;

      if (Z < min_z_) {
          continue; 
      }

      // --- 核心投影公式 (3D -> 2D) ---
      int u = static_cast<int>((fx_ * X / Z) + cx_); 
      int v = static_cast<int>((fy_ * Y / Z) + cy_);
      bool is_inside = (u >= 0 && u < image_width_ && v >= 0 && v < image_height_);

      if (is_inside) {
        // 在范围内：绘制
        cv::circle(image, cv::Point(u, v), radius_, cv::Scalar(255, 255, 255), -1);
      } else {
        // 越界日志，使用 Throttle 机制防止刷屏
        // 参数说明：logger, clock, 毫秒数, 格式化字符串...
        // 这里设置为 2000ms (2秒) 打印一次，既能看到提示，又不至于眼花
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "Point out of bounds: (u=%d, v=%d) | Depth Z=%.2f", u, v, Z);
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
  double fx_, fy_, cx_, cy_;
  int radius_;
  double min_z_;
  int image_width_, image_height_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageViewerTalker>());
  rclcpp::shutdown();
  return 0;
}
