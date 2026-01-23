#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <thread>
#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "object3d_msgs/msg/object3_d_array.hpp"

using namespace std::chrono_literals;

// 滤波状态结构体
struct FilterState {
    double u = 0.0; 
    double v = 0.0; 
    double r = 0.0; 
    bool active = false; 
    int missed_frames = 0; 
};

class ImageViewerTalker : public rclcpp::Node
{
public:
  ImageViewerTalker()
  : Node("image_viewer_talker"), window_initialized_(false)
  {
    // 1. 参数声明 (完全保持你原来的参数)
    this->declare_parameter<double>("fx", 1612.8);
    this->declare_parameter<double>("fy", 1612.8);
    this->declare_parameter<double>("cx", 640.0);
    this->declare_parameter<double>("cy", 360.0);
    this->declare_parameter<double>("min_z_threshold", 0.3);
    this->declare_parameter<int>("projection_width", 1280);
    this->declare_parameter<int>("projection_height", 720);

    // 新增：滤波参数 (调节这两个数改变平滑程度)
    this->declare_parameter<double>("filter_min_alpha", 0.5); // 越小越稳
    this->declare_parameter<double>("filter_gamma", 0.1);     // 越大跟手越快

    fx_ = this->get_parameter("fx").as_double();
    fy_ = this->get_parameter("fy").as_double();
    cx_ = this->get_parameter("cx").as_double();
    cy_ = this->get_parameter("cy").as_double();
    min_z_ = this->get_parameter("min_z_threshold").as_double();
    projection_width_ = this->get_parameter("projection_width").as_int();
    projection_height_ = this->get_parameter("projection_height").as_int();

    filter_min_alpha_ = this->get_parameter("filter_min_alpha").as_double();
    filter_gamma_ = this->get_parameter("filter_gamma").as_double();

    // 2. QOS 设置
    auto qos = rclcpp::SensorDataQoS().keep_last(1);

    // 3. 创建发布者
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("projection_image_topic", qos);
    
    // 4. 创建订阅者 (改回你原来的 topic 名字！)
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
  // 自适应滤波函数：diff越大(运动越快)，alpha越大(更新越快)
  double adaptive_filter(double current, double prev, double diff) {
      double norm_diff = std::min(diff / 10.0, 1.0);
      double alpha = filter_min_alpha_ + filter_gamma_ * norm_diff;
      if (alpha > 1.0) alpha = 1.0;
      return prev * (1.0 - alpha) + current * alpha;
  }

  void points_callback(const object3d_msgs::msg::Object3DArray::SharedPtr msg)
  {
    // 1. 创建黑色画布
    cv::Mat image = cv::Mat::zeros(projection_height_, projection_width_, CV_8UC3);

    // --- 滤波准备：标记所有旧目标为不活跃 ---
    for (auto& pair : tracked_objects_) {
        pair.second.active = false;
    }

    // 2. 处理数据
    for (const auto & obj : msg->objects)
    {
      double Z = obj.point.z;
      if (Z < min_z_) continue; 

      // === [保留原逻辑] 计算原始坐标 ===
      // 这里完全是你原来的公式，保证坐标系统一致
      double raw_u = (fx_ * obj.point.x / Z) + cx_; 
      double raw_v = (fy_ * obj.point.y / Z) + cy_;
      
      double physical_radius = std::min(obj.width_m, obj.height_m)/2.0; 
      double raw_r = (physical_radius * fx_) / Z;
      if (raw_r < 5.0) raw_r = 5.0;

      // === [新增] 滤波逻辑 ===
      
      // A. 寻找匹配的旧目标 (简单的距离匹配)
      int best_id = -1;
      double min_dist = 200.0; // 像素阈值，超过这个距离认为是新物体

      for (auto& pair : tracked_objects_) {
          if (pair.second.active) continue; // 已经被匹配过了
          double dist = std::sqrt(std::pow(raw_u - pair.second.u, 2) + std::pow(raw_v - pair.second.v, 2));
          if (dist < min_dist) {
              min_dist = dist;
              best_id = pair.first;
          }
      }

      if (best_id != -1) {
          // B. 找到了上一帧的自己 -> 进行滤波
          FilterState& state = tracked_objects_[best_id];
          
          double diff_u = raw_u - state.u;
          double diff_v = raw_v - state.v;
          double motion = std::sqrt(diff_u*diff_u + diff_v*diff_v); // 运动幅度

          state.u = adaptive_filter(raw_u, state.u, motion);
          state.v = adaptive_filter(raw_v, state.v, motion);
          state.r = adaptive_filter(raw_r, state.r, std::abs(raw_r - state.r));
          
          state.active = true;
          state.missed_frames = 0;
      } else {
          // C. 没找到 -> 新目标，直接赋值
          int new_id = 0;
          while (tracked_objects_.find(new_id) != tracked_objects_.end()) new_id++;
          
          FilterState new_state;
          new_state.u = raw_u;
          new_state.v = raw_v;
          new_state.r = raw_r;
          new_state.active = true;
          tracked_objects_[new_id] = new_state;
      }
    }

    // 3. 绘制逻辑 (遍历 active 或者刚消失不久的物体)
    for (auto it = tracked_objects_.begin(); it != tracked_objects_.end(); ) {
        FilterState& state = it->second;

        // 如果不活跃，增加消失计数
        if (!state.active) {
            state.missed_frames++;
            // 只有连续丢帧超过 5 帧才删除，这能防止检测闪烁
            if (state.missed_frames > 5) {
                it = tracked_objects_.erase(it);
                continue;
            }
        }

        // 只要还“存活”就画出来
        int u = static_cast<int>(state.u);
        int v = static_cast<int>(state.v);
        int radius = static_cast<int>(state.r);

        if (u >= 0 && u < projection_width_ && v >= 0 && v < projection_height_) {
            cv::circle(image, cv::Point(u, v), radius/2, cv::Scalar(0, 255, 0), -1);// 填充圆（缩小一倍）
        }
        
        ++it;
    }

    // 4. 显示逻辑
    if (!image.empty()) {
        if (!window_initialized_) {
            cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
            cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            window_initialized_ = true;
        }
        
        cv::imshow(window_name_, image);
        cv::waitKey(1); 
    }

    // 5. 发布图像
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
  
  // 滤波参数
  double filter_min_alpha_;
  double filter_gamma_;
  std::map<int, FilterState> tracked_objects_; // 跟踪列表

  std::string window_name_;
  bool window_initialized_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageViewerTalker>());
  rclcpp::shutdown();
  return 0;
}