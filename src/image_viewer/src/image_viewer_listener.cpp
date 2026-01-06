#include <memory>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

class ImageSubscriber : public rclcpp::Node
{
public:
  ImageSubscriber() : Node("image_viewer_listener")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "image_topic", 10,
      std::bind(&ImageSubscriber::image_callback, this, std::placeholders::_1));
    // 设置窗口为全屏
    cv::namedWindow("Received Image", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Received Image", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try
    {
      // 转换ROS图像消息到OpenCV格式
      cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;

      // 显示图像
      cv::imshow("Received Image", image);
      cv::waitKey(1);  // 必须调用，处理窗口事件
    }
    catch (cv_bridge::Exception & e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageSubscriber>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
