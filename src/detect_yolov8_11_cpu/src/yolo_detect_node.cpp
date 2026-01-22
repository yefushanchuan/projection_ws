#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>

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

// Local Lib (CPU Detector)
#include "detect_yolov8_11_cpu/cpu_detect.hpp"

// Yolo Common Lib (统一工具库)
#include "yolo_common/visualization.hpp"
#include "yolo_common/math_utils.hpp"
#include "yolo_common/ros_utils.hpp" 
#include "yolo_common/file_utils.hpp"

using namespace std::chrono_literals;

class YoloDetectNode : public rclcpp::Node {
public:
    YoloDetectNode() : Node("yolo_detect_node") {
        // ============================================================
        // 1. 声明与读取参数
        // ============================================================
        this->declare_parameter("camera.fx", 905.5593);
        this->declare_parameter("camera.fy", 905.5208);
        this->declare_parameter("camera.cx", 663.4498);
        this->declare_parameter("camera.cy", 366.7621);
        this->declare_parameter("conf_thres", 0.50);
        this->declare_parameter("nms_thres", 0.45);
        this->declare_parameter("show_image", true);
        this->declare_parameter("model_filename", "yolo11n.onnx"); 
        this->declare_parameter("class_labels_file", ""); 

        // 读取相机内参到统一结构体
        cam_param_.fx = this->get_parameter("camera.fx").as_double();
        cam_param_.fy = this->get_parameter("camera.fy").as_double();
        cam_param_.cx = this->get_parameter("camera.cx").as_double();
        cam_param_.cy = this->get_parameter("camera.cy").as_double();
        
        // 读取初始配置
        double conf_thres = this->get_parameter("conf_thres").as_double();
        double nms_thres = this->get_parameter("nms_thres").as_double();
        show_image_ = this->get_parameter("show_image").as_bool();
        
        // ============================================================
        // 2. 初始化检测器 (Unified Path Logic)
        // ============================================================
        std::string model_filename = this->get_parameter("model_filename").as_string();
        std::string class_filename = this->get_parameter("class_labels_file").as_string();
        
        std::filesystem::path ws_root;
        bool found_ws = false;

        // --- 2.1 查找工作空间根目录 ---
        try {
            std::filesystem::path p(ament_index_cpp::get_package_share_directory("detect_yolov8_11_cpu"));
            while (p.has_parent_path() && p.filename() != "install") {
                p = p.parent_path();
            }
            if (p.filename() == "install") {
                ws_root = p.parent_path();
                found_ws = true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to locate 'install' dir.");
                return;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Path error: %s", e.what());
            return;
        }

        // --- 2.2 解析并加载模型 (Workspace/models) ---
        if (model_filename.front() == '/') {
            resolved_model_path_ = model_filename;
        } else {
            resolved_model_path_ = (ws_root / "models" / model_filename).string();
        }

        if (!std::filesystem::exists(resolved_model_path_)) {
            RCLCPP_ERROR(this->get_logger(), "Model not found: %s", resolved_model_path_.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loading model: %s", resolved_model_path_.c_str());

        try {
            // 实例化 (内部默认已经是 COCO)
            detector_ = std::make_unique<CPU_Detect>(resolved_model_path_);
            
            // --- 2.3 尝试加载自定义类别 (Workspace/configs) ---
            // 只有当参数不为空时，才尝试覆盖默认值
            if (!class_filename.empty()) {
                std::string final_class_path;
                if (class_filename.front() == '/') {
                    final_class_path = class_filename;
                } else {
                    final_class_path = (ws_root / "configs" / class_filename).string();
                }

                // 只有文件存在且解析出内容，才进行覆盖
                if (std::filesystem::exists(final_class_path)) {
                    auto custom_classes = yolo_common::utils::LoadClassesFromFile(final_class_path);
                    if (!custom_classes.empty()) {
                        detector_->config_.class_names = custom_classes;
                        // 别忘了更新 class_num，推理循环通常依赖这个
                        detector_->config_.class_num = custom_classes.size();
                        RCLCPP_INFO(this->get_logger(), "Loaded custom classes: %s", class_filename.c_str());
                    }
                }
            }
            
            // --- 2.4 应用阈值 ---
            detector_->config_.conf_thres = (float)conf_thres;
            detector_->config_.nms_iou_thres = (float)nms_thres;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Init failed: %s", e.what());
            return;
        }

        // ============================================================
        // 3. 设置 ROS 通信
        // ============================================================
        
        // 动态参数回调
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&YoloDetectNode::parametersCallback, this, std::placeholders::_1));

        // QoS 设置 (SensorData 模式，低延迟)
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile), qos_profile);

        // 订阅图像
        color_sub_.subscribe(this, "/camera/realsense_d435i/color/image_raw", qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, "/camera/realsense_d435i/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile());

        // 时间同步
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), color_sub_, depth_sub_);
        sync_->registerCallback(std::bind(&YoloDetectNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));
        // 允许一定的同步误差 (0.1s)
        sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));

        // 发布结果
        publisher_ = this->create_publisher<object3d_msgs::msg::Object3DArray>("target_points_array", qos);
        
        RCLCPP_INFO(this->get_logger(), "YoloDetectNode Initialized successfully.");
    }

private:
    // ============================================================
    // 动态参数回调函数
    // ============================================================
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";

        for (const auto &param : parameters) {
            if (param.get_name() == "show_image") {
                show_image_ = param.as_bool();
            } 
            else if (param.get_name() == "conf_thres") {
                if (detector_) {
                    detector_->config_.conf_thres = (float)param.as_double();
                    RCLCPP_INFO(this->get_logger(), "Updated conf_thres: %.2f", detector_->config_.conf_thres);
                }
            }
            else if (param.get_name() == "nms_thres") {
                if (detector_) {
                    detector_->config_.nms_iou_thres = (float)param.as_double();
                    RCLCPP_INFO(this->get_logger(), "Updated nms_thres: %.2f", detector_->config_.nms_iou_thres);
                }
            }
        }
        return result;
    }

    // ============================================================
    // 核心处理回调 (图像同步)
    // ============================================================
    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_color, 
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth) {
        // 1. FPS 计数更新
        fps_monitor_.Tick();

        // 2. 图像转换 (ROS -> OpenCV)
        cv::Mat color_img, depth_img;
        try {
            color_img = cv_bridge::toCvShare(msg_color, "bgr8")->image;
            // 深度图不拷贝，只读引用，提高效率
            depth_img = cv_bridge::toCvShare(msg_depth, "16UC1")->image; 
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // 3. 模型推理
        // 注意：推理只用彩色图，返回统一结果结构
        auto results = detector_->detect(color_img);

        // 4. 消息生成与发布 (核心优化点：使用 ros_utils 一行转换)
        if (publisher_->get_subscription_count() > 0 && !results.empty()) {
            object3d_msgs::msg::Object3DArray array_msg;
            
            yolo_common::ros_utils::ResultsTo3DMessage(
                results, 
                depth_img, 
                msg_color->header, 
                cam_param_, 
                array_msg
            );
            
            publisher_->publish(array_msg);
        }

        // 5. 可视化窗口管理
        if (show_image_) {
            // ShowWindow 内部处理绘制和 FPS 显示
            yolo_common::vis::ShowWindow("CPU Detect", color_img, results, win_created_flag_, fps_monitor_.Get());
        } else if (win_created_flag_) {
            // 参数关闭显示时，自动销毁窗口
            cv::destroyWindow("CPU Detect");
            win_created_flag_ = false;
            cv::waitKey(1);
        }
    }

    // ============================================================
    // 成员变量
    // ============================================================
    std::string resolved_model_path_;

    // 统一内参结构
    yolo_common::ros_utils::CameraIntrinsics cam_param_;
    
    // FPS 监控器
    yolo_common::utils::FpsMonitor fps_monitor_;
    
    // 配置与逻辑
    bool show_image_;
    std::unique_ptr<CPU_Detect> detector_;
    
    // ROS 通信对象
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr publisher_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    
    // GUI 状态
    bool win_created_flag_ = false;
};

// ============================================================
// Main 入口
// ============================================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloDetectNode>());
    rclcpp::shutdown();
    return 0;
}