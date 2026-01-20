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

// Local Lib
#include "detect_yolov8_11_cpu/cpu_detect.hpp"

// Yolo Common Lib
#include "yolo_common/visualization.hpp"
#include "yolo_common/math_utils.hpp"

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
            detector_ = std::make_unique<CPU_Detect>(final_model_path, conf_thres_);
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
                RCLCPP_INFO(this->get_logger(), "Update show_image: %s", show_image_ ? "true" : "false");
            }
        }
        return result;
    }

    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_color, 
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth) {
        // FPS Calc
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

        // === 1. 推理 (返回 UnifiedResult) ===
        // 注意：推理引擎不修改 color_img，也不负责显示
        auto results = detector_->detect(color_img);

        // === 2. 可视化管理 (调用 yolo_common) ===
        if (show_image_) {
            // ShowWindow 内部有 waitKey(1)，且绘制 FPS 和结果
            yolo_common::vis::ShowWindow("Detection Result", color_img, results, win_created_flag_, fps_);
        } else {
            // 如果需要关闭窗口
            if (win_created_flag_) {
                try {
                    cv::destroyWindow("Detection Result");
                } catch(...) {}
                win_created_flag_ = false;
                cv::waitKey(1); // 必须调用，否则窗口不会消失
            }
        }

        if (publisher_->get_subscription_count() == 0 || results.empty()) return;

        object3d_msgs::msg::Object3DArray array_msg;
        array_msg.header = msg_color->header;

        for (auto& res : results) {
            // 使用 common math 提取深度
            float d = yolo_common::math::GetRobustDepth(depth_img, res.center);
            
            if (d <= 0) continue;

            float Z = d; //GetRobustDepth 已经返回米
            float X = (res.center.x - cx_) * Z / fx_;
            float Y = (res.center.y - cy_) * Z / fy_;

            object3d_msgs::msg::Object3D obj;
            obj.point.x = X;
            obj.point.y = Y;
            obj.point.z = Z;
            obj.width_m = (res.box.width * Z) / fx_;
            obj.height_m = (res.box.height * Z) / fy_;
            
            obj.class_name = res.class_name;
            obj.score = res.score;

            array_msg.objects.push_back(obj);
        }

        if (!array_msg.objects.empty()) {
            publisher_->publish(array_msg);
        }
    }

    double fx_, fy_, cx_, cy_;
    double conf_thres_;
    bool show_image_;
    std::unique_ptr<CPU_Detect> detector_;
    
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr publisher_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
    int frame_count_ = 0;
    double fps_ = 0.0;
    
    // 窗口状态 (供 ShowWindow 使用)
    bool win_created_flag_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloDetectNode>());
    rclcpp::shutdown();
    return 0;
}