#include <memory>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

class ImageViewerSubscriber : public rclcpp::Node
{
public:
  ImageViewerSubscriber() : Node("image_viewer_listener")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "image_topic", 10,
      std::bind(&ImageViewerSubscriber::image_callback, this, std::placeholders::_1));
      
    // 初始化窗口
    window_name_ = "Received Image";
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    
    RCLCPP_INFO(this->get_logger(), "ImageViewer Listener started (Fullscreen Mode).");
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try
    {
      // 零拷贝转换 (toCvShare 更高效)
      cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;

      // 只有图像非空才显示
      if (!image.empty()) {
          cv::imshow(window_name_, image);
          cv::waitKey(1); 
      }
    }
    catch (cv_bridge::Exception & e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  std::string window_name_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageViewerSubscriber>();
  rclcpp::spin(node);
  
  // 【核心修正】节点退出时，强制销毁窗口
  cv::destroyAllWindows();
  
  rclcpp::shutdown();
  return 0;
}