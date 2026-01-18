#include <chrono>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <cstdlib> 

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
    // 1. 参数声明
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

    window_name_ = "projection Image";
  }

  ~ImageViewerTalker() {
      cv::destroyAllWindows();
  }

private:
  void points_callback(const object3d_msgs::msg::Object3DArray::SharedPtr msg)
  {
    // 1. 创建黑色画布
    cv::Mat image = cv::Mat::zeros(projection_height_, projection_width_, CV_8UC3);

    // 2. 绘制光斑
    for (const auto & obj : msg->objects)
    {
      double Z = obj.point.z;
      if (Z < min_z_) continue; 

      int u = static_cast<int>((fx_ * obj.point.x / Z) + cx_); 
      int v = static_cast<int>((fy_ * obj.point.y / Z) + cy_);
      
      double physical_radius = std::min(obj.width_m, obj.height_m)/2.0; 
      int radius_pixel = static_cast<int>((physical_radius * fx_) / Z);
      
      if (radius_pixel < 5) radius_pixel = 5;

      if (u >= 0 && u < projection_width_ && v >= 0 && v < projection_height_) {
        // 画实心绿圆
        cv::circle(image, cv::Point(u, v), radius_pixel, cv::Scalar(0, 255, 0), -1);
      }
    }

    // 3. 显示逻辑
    if (!image.empty()) {
        if (!window_initialized_) {
            // 第一次：初始化全屏
            cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
            cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            window_initialized_ = true;
        }
        
        cv::imshow(window_name_, image);
        cv::waitKey(1); // 必须调用，且只等 1ms

        if (wmctrl_counter_ < 10) {
            // 构造命令字符串
            // -r : 指定窗口名称 (必须和 cv::namedWindow 的名字完全一致)
            // -b add,fullscreen,above : 添加 "全屏" 和 "置顶" 属性
            std::string cmd = "wmctrl -r '" + window_name_ + "' -b add,above";
            
            int ret = std::system(cmd.c_str());
            
            // 只有当 wmctrl 成功找到窗口并执行后，计数器才增加
            // 这样可以防止窗口还没弹出来时计数器就跑完了
            if (ret == 0) {
                wmctrl_counter_++;
            }
        }
    }

    // 4. 发布图像
    std_msgs::msg::Header header;
    header.stamp = this->now();
    header.frame_id = "projector_frame";
    try {
      image_pub_->publish(*cv_bridge::CvImage(header, "bgr8", image).toImageMsg());
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "CvImage error: %s", e.what());
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Subscription<object3d_msgs::msg::Object3DArray>::SharedPtr points_sub_;
  
  double fx_, fy_, cx_, cy_;
  double min_z_;
  int projection_width_, projection_height_;
  
  std::string window_name_;
  bool window_initialized_;
  int wmctrl_counter_ = 0; 
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageViewerTalker>());
  rclcpp::shutdown();
  return 0;
}