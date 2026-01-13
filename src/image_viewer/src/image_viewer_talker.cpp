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
  : Node("image_viewer_talker")
  {
    this->declare_parameter<double>("fx", 1612.8); // 注意：这里通常需要根据投影仪标定
    this->declare_parameter<double>("fy", 1612.8);
    this->declare_parameter<double>("cx", 640.0); 
    this->declare_parameter<double>("cy", 360.0);
    this->declare_parameter<int>("point_radius", 15);
    this->declare_parameter<double>("min_z_threshold", 0.2); // 最小深度阈值
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

    // 发布生成的黑底打点图
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("projection_image_topic", 10);

    // 订阅经过坐标转换后的 3D 点
    points_sub_  = this->create_subscription<object3d_msgs::msg::Object3DArray>(
      "target_points_projection_array", 10,
      std::bind(&ImageViewerTalker::points_callback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "Image Viewer Talker Started. Canvas: %dx%d", image_width_, image_height_);
  }

private:
  void points_callback(const object3d_msgs::msg::Object3DArray::SharedPtr msg)
  {
    cv::Mat image = cv::Mat::zeros(image_height_, image_width_, CV_8UC3);

    for (const auto & obj : msg->objects)
    {
      double X = obj.point.x;
      double Y = obj.point.y;
      double Z = obj.point.z;

      if (Z < min_z_) continue; 

      // 1. 计算投影中心坐标 (u, v)
      int u = static_cast<int>((fx_ * X / Z) + cx_); 
      int v = static_cast<int>((fy_ * Y / Z) + cy_);
      
      // 2. === 核心修改：计算投影半径 ===
      // 从消息里拿出物体的物理宽高
      double obj_w = obj.width_m;
      double obj_h = obj.height_m;
      
      // 逻辑：取最大内切圆 -> 直径取短边 -> 半径 = 短边 / 2
      double physical_radius = std::min(obj_w, obj_h) / 2.0;
      
      // 反算：这个物理半径在投影仪画面上占多少像素？
      // 像素半径 = (物理半径 * 投影仪焦距) / 投影距离Z
      // 这里的 fx_ 是投影仪的 fx (1612.8)
      int proj_radius_pixel = static_cast<int>((physical_radius * fx_) / Z);
      
      // 限制最小半径，防止太远看不见
      if (proj_radius_pixel < 5) proj_radius_pixel = 5;
      // ==============================

      bool is_inside = (u >= 0 && u < image_width_ && v >= 0 && v < image_height_);

      if (is_inside) {
        // 使用算出来的动态半径画圆
        cv::circle(image, cv::Point(u, v), proj_radius_pixel/2, cv::Scalar(0, 255, 0), -1);
      } 
    }

    // 5. 发布图像
    std_msgs::msg::Header header;
    header.stamp = this->now();
    header.frame_id = "projector_frame"; // 或者 camera_frame

    try {
      auto ros_image = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
      image_pub_->publish(*ros_image);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "CvImage conversion error: %s", e.what());
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<object3d_msgs::msg::Object3DArray>::SharedPtr points_sub_;
  
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