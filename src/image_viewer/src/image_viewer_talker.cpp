#include <chrono>
#include <memory>
#include <string>
#include <cstdlib>
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
  : Node("image_viewer_talker"), window_initialized_(false)
  {
    // 1. 参数声明与获取
    this->declare_parameter<double>("fx", 1612.8);
    this->declare_parameter<double>("fy", 1612.8);
    this->declare_parameter<double>("cx", 640.0);
    this->declare_parameter<double>("cy", 360.0);
    this->declare_parameter<double>("min_z_threshold", 0.5);
    this->declare_parameter<int>("projection_width", 1280);
    this->declare_parameter<int>("projection_height", 720);

    fx_ = this->get_parameter("fx").as_double();
    fy_ = this->get_parameter("fy").as_double();
    cx_ = this->get_parameter("cx").as_double();
    cy_ = this->get_parameter("cy").as_double();
    min_z_ = this->get_parameter("min_z_threshold").as_double();
    projection_width_ = this->get_parameter("projection_width").as_int();
    projection_height_ = this->get_parameter("projection_height").as_int();

    // 2. QOS 设置
    auto qos = rclcpp::SensorDataQoS().keep_last(1);

    // 3. 创建发布者和订阅者
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("projection_image_topic", qos);
    
    points_sub_  = this->create_subscription<object3d_msgs::msg::Object3DArray>(
      "target_points_projection_array", qos,
      std::bind(&ImageViewerTalker::points_callback, this, std::placeholders::_1)
    );

    // 设置窗口名称
    window_name_ = "projection Image";
  }

  ~ImageViewerTalker() {
      if (window_initialized_) {
          cv::destroyAllWindows();
      }
  }

private:
  void points_callback(const object3d_msgs::msg::Object3DArray::SharedPtr msg)
  {
    // 1. 先生成图像数据（消息转换与绘图）
    cv::Mat image = cv::Mat::zeros(projection_height_, projection_width_, CV_8UC3);

    for (const auto & obj : msg->objects)
    {
      double Z = obj.point.z;
      if (Z < min_z_) continue; 

      int u = static_cast<int>((fx_ * obj.point.x / Z) + cx_); 
      int v = static_cast<int>((fy_ * obj.point.y / Z) + cy_);
      
      double physical_radius = std::min(obj.width_m, obj.height_m) / 2.0;
      int proj_radius_pixel = static_cast<int>((physical_radius * fx_) / Z);
      if (proj_radius_pixel < 5) proj_radius_pixel = 5;

      if (u >= 0 && u < projection_width_ && v >= 0 && v < projection_height_) {
        cv::circle(image, cv::Point(u, v), proj_radius_pixel/2, cv::Scalar(0, 255, 0), -1);
      }
    }

    // 2. 懒加载：如果是第一次收到数据，才创建窗口
    if (!window_initialized_) {
        init_window_once();
    }

    // 3. 显示当前帧
    if (!image.empty()) {
        cv::imshow(window_name_, image);
        cv::waitKey(1); 
    }

    // 4. 发布图像消息（此时已经完成了绘制和显示）
    std_msgs::msg::Header header;
    header.stamp = this->now();
    header.frame_id = "projector_frame";
    try {
      image_pub_->publish(*cv_bridge::CvImage(header, "bgr8", image).toImageMsg());
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "CvImage conversion error: %s", e.what());
    }
  }

  // 辅助函数：只在第一次收到数据时执行一次
  void init_window_once() {
      RCLCPP_INFO(this->get_logger(), "Data received. Initializing Projection Window...");
      
      // 创建窗口并设置全屏
      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
      cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
      
      // 必须先 imshow 一次，wmctrl 才能找到窗口句柄
      // 这里我们可以直接显示一张临时的黑底或直接利用 callback 里即将显示的图
      // 为了稳妥，先显示个黑底占位，紧接着 callback 后面会刷上真正的图
      cv::imshow(window_name_, cv::Mat::zeros(projection_height_, projection_width_, CV_8UC3));
      cv::waitKey(50); // 给系统一点时间注册窗口

      // 调用 wmctrl -a 激活窗口到最前
      std::string cmd = "wmctrl -a \"" + window_name_ + "\"";
      if (std::system(cmd.c_str()) != 0) {
          RCLCPP_WARN(this->get_logger(), "wmctrl command failed. (Is wmctrl installed?)");
      } else {
          RCLCPP_INFO(this->get_logger(), "Window activated.");
      }

      window_initialized_ = true;
  }

  // 成员变量
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<object3d_msgs::msg::Object3DArray>::SharedPtr points_sub_;
  
  double fx_, fy_, cx_, cy_;
  double min_z_;
  int projection_width_, projection_height_;
  
  std::string window_name_;
  bool window_initialized_; // 新增：窗口初始化状态标志
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageViewerTalker>());
  rclcpp::shutdown();
  return 0;
}