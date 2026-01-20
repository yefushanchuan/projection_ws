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
#include "object3d_msgs/msg/object3_d_array.hpp"

// Local Lib
#include "detect_yolov8_11_cpu/cpu_detect.hpp"

// Yolo Common Lib
#include "yolo_common/visualization.hpp"
#include "yolo_common/math_utils.hpp"
#include "yolo_common/ros_utils.hpp" // 引入ROS工具库

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

        // 统一参数结构
        cam_param_.fx = this->get_parameter("camera.fx").as_double();
        cam_param_.fy = this->get_parameter("camera.fy").as_double();
        cam_param_.cx = this->get_parameter("camera.cx").as_double();
        cam_param_.cy = this->get_parameter("camera.cy").as_double();
        
        double conf_thres = this->get_parameter("conf_thres").as_double();
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
                std::string package_share_directory = ament_index_cpp::get_package_share_directory("detect_yolov8_11_cpu");
                final_model_path = (fs::path(package_share_directory) / "models" / model_filename).string();
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Error finding package path: %s", e.what());
                return;
            }
        }

        if (!fs::exists(final_model_path)) {
            RCLCPP_ERROR(this->get_logger(), "Model not found: %s", final_model_path.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Initializing CPU_Detect with: %s", final_model_path.c_str());
        
        try {
            detector_ = std::make_unique<CPU_Detect>(final_model_path, (float)conf_thres);
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
        sync_->registerCallback(std::bind(&YoloDetectNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));
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
                // 窗口销毁逻辑交由主循环处理
            }
        }
        return result;
    }

    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_color, 
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth) {
        // 1. FPS 更新
        fps_monitor_.Tick();

        cv::Mat color_img, depth_img;
        try {
            color_img = cv_bridge::toCvShare(msg_color, "bgr8")->image;
            depth_img = cv_bridge::toCvShare(msg_depth, "16UC1")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        // 2. 推理 (返回 UnifiedResult)
        // 注意：推理引擎不修改 color_img
        auto results = detector_->detect(color_img);

        // 3. 消息转换与发布 (优化点：替换原有的几十行循环)
        if (publisher_->get_subscription_count() > 0 && !results.empty()) {
            object3d_msgs::msg::Object3DArray array_msg;
            
            yolo_common::ros_utils::ResultsTo3DMessage(
                results, depth_img, msg_color->header, cam_param_, array_msg
            );
            
            publisher_->publish(array_msg);
        }

        // 4. 可视化管理
        if (show_image_) {
            yolo_common::vis::ShowWindow("Detection Result", color_img, results, win_created_flag_, fps_monitor_.Get());
        } else if (win_created_flag_) {
            cv::destroyWindow("Detection Result");
            win_created_flag_ = false;
            cv::waitKey(1);
        }
    }

    // 核心参数
    yolo_common::ros_utils::CameraIntrinsics cam_param_;
    yolo_common::utils::FpsMonitor fps_monitor_;
    
    bool show_image_;
    std::unique_ptr<CPU_Detect> detector_;
    
    // ROS 通信
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr publisher_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    
    // 窗口状态
    bool win_created_flag_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloDetectNode>());
    rclcpp::shutdown();
    return 0;
}