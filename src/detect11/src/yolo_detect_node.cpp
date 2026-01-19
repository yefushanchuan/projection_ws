#include <chrono>
#include <filesystem>
#include <iostream>

// ROS2 Includes
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "ament_index_cpp/get_package_share_directory.hpp" 

// Custom Message
#include "object3d_msgs/msg/object3_d.hpp"
#include "object3d_msgs/msg/object3_d_array.hpp"

// Include our new detector library
#include "detect11/cpu_detect.hpp"

using namespace std::chrono_literals;

class YoloDetectNode : public rclcpp::Node {
public:
    YoloDetectNode() : Node("yolo_detect_node") {
        // 1. 声明参数
        this->declare_parameter("camera.fx", 905.5593);
        this->declare_parameter("camera.fy", 905.5208);
        this->declare_parameter("camera.cx", 663.4498);
        this->declare_parameter("camera.cy", 366.7621);
        this->declare_parameter("conf_thres", 0.50);
        this->declare_parameter("show_image", true);
        this->declare_parameter("model_filename", "yolo11n.onnx"); 

        fx_ = this->get_parameter("camera.fx").as_double();
        fy_ = this->get_parameter("camera.fy").as_double();
        cx_ = this->get_parameter("camera.cx").as_double();
        cy_ = this->get_parameter("camera.cy").as_double();
        conf_thres_ = this->get_parameter("conf_thres").as_double();
        show_image_ = this->get_parameter("show_image").as_bool();
        
        // 2. 加载模型逻辑
        std::string model_filename = this->get_parameter("model_filename").as_string();
        std::string final_model_path;
        namespace fs = std::filesystem;
        fs::path p(model_filename);

        if (p.is_absolute()) {
            final_model_path = model_filename;
        } else {
            try {
                std::string package_share_directory = ament_index_cpp::get_package_share_directory("detect11");
                final_model_path = (fs::path(package_share_directory) / "models" / model_filename).string();
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Package error: %s", e.what());
                return;
            }
        }

        if (!fs::exists(final_model_path)) {
            RCLCPP_ERROR(this->get_logger(), "Model not found: %s", final_model_path.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Initializing Yolo11Detector with: %s", final_model_path.c_str());
        
        try {
            // 初始化探测器类
            detector_ = std::make_unique<Yolo11Detector>(final_model_path, conf_thres_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Detector init failed: %s", e.what());
            return;
        }

        // 3. 订阅与发布
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile), qos_profile);

        color_sub_.subscribe(this, "/camera/realsense_d435i/color/image_raw", qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, "/camera/realsense_d435i/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), color_sub_, depth_sub_);
        sync_->registerCallback(&YoloDetectNode::sync_callback, this);
        sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));

        publisher_ = this->create_publisher<object3d_msgs::msg::Object3DArray>("target_points_array", qos);
        
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&YoloDetectNode::parametersCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Node Ready.");
    }

private:
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &param : parameters) {
            if (param.get_name() == "show_image" && param.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) {
                show_image_ = param.as_bool();
                RCLCPP_INFO(this->get_logger(), "Update show_image: %s", show_image_ ? "true" : "false");
            }
        }
        return result;
    }

    float get_robust_depth(const cv::Mat& depth_img, int cx, int cy) {
        if (cx < 0 || cx >= depth_img.cols || cy < 0 || cy >= depth_img.rows) return -1.0;
        int x_min = std::max(0, cx - 2);
        int x_max = std::min(depth_img.cols, cx + 3);
        int y_min = std::max(0, cy - 2);
        int y_max = std::min(depth_img.rows, cy + 3);

        std::vector<unsigned short> valid_depths;
        valid_depths.reserve(25);
        for (int y = y_min; y < y_max; ++y) {
            for (int x = x_min; x < x_max; ++x) {
                unsigned short d = depth_img.at<unsigned short>(y, x);
                if (d > 0) valid_depths.push_back(d);
            }
        }
        if (valid_depths.empty()) return -1.0;
        size_t n = valid_depths.size() / 2;
        std::nth_element(valid_depths.begin(), valid_depths.begin() + n, valid_depths.end());
        return (float)valid_depths[n];
    }

    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_color, 
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth) {
        // FPS
        auto now = std::chrono::steady_clock::now();
        frame_count_++;
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
        if (elapsed >= 1.0) {
            fps_ = frame_count_ / elapsed;
            frame_count_ = 0;
            start_time_ = now;
        }

        cv::Mat color_img, depth_img;
        try {
            color_img = cv_bridge::toCvShare(msg_color, "bgr8")->image;
            depth_img = cv_bridge::toCvShare(msg_depth, "16UC1")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        if (show_image_) {
            cv::putText(color_img, "FPS: " + std::to_string((int)fps_), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }

        // === 调用分离出去的推理引擎 ===
        auto detections = detector_->detect(color_img, show_image_);
        // ============================

        // 显示逻辑
        if (show_image_) {
            const std::string win_name = "YOLOv11 Detection";
            cv::namedWindow(win_name, cv::WINDOW_NORMAL);
            cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            cv::imshow(win_name, color_img);
            cv::waitKey(1);
        } else {
            try {
                if (cv::getWindowProperty("YOLOv11 Detection", cv::WND_PROP_VISIBLE) >= 0) {
                    cv::destroyWindow("YOLOv11 Detection");
                    cv::waitKey(1);
                }
            } catch (...) {}
        }

        if (publisher_->get_subscription_count() == 0 || detections.empty()) return;

        object3d_msgs::msg::Object3DArray array_msg;
        array_msg.header = msg_color->header;

        for (const auto& det : detections) {
            float d = get_robust_depth(depth_img, det.center.x, det.center.y);
            if (d <= 0) continue;

            float Z = d / 1000.0f;
            float X = (det.center.x - cx_) * Z / fx_;
            float Y = (det.center.y - cy_) * Z / fy_;

            object3d_msgs::msg::Object3D obj;
            obj.point.x = X;
            obj.point.y = Y;
            obj.point.z = Z;
            obj.width_m = (det.box.width * Z) / fx_;
            obj.height_m = (det.box.height * Z) / fy_;
            
            // 使用辅助函数获取类别名
            obj.class_name = detector_->getClassName(det.class_id);
            obj.score = det.confidence;

            array_msg.objects.push_back(obj);
        }

        if (!array_msg.objects.empty()) {
            publisher_->publish(array_msg);
        }
    }

    double fx_, fy_, cx_, cy_;
    double conf_thres_;
    bool show_image_;
    std::unique_ptr<Yolo11Detector> detector_; // 仅持有指针
    
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr publisher_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
    int frame_count_ = 0;
    double fps_ = 0.0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloDetectNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}