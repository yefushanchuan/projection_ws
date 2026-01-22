#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include <chrono>
#include <filesystem>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "object3d_msgs/msg/object3_d_array.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "segment_yolov8_11_bpu/bpu_seg_dnn.hpp"

// 引入 yolo_common 头文件
#include "yolo_common/visualization.hpp"
#include "yolo_common/math_utils.hpp"
#include "yolo_common/ros_utils.hpp" 

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNodeOutput;

class YoloSegNode : public hobot::dnn_node::DnnNode {
public:
    YoloSegNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : hobot::dnn_node::DnnNode(node_name, options) {
        
        // 1. 参数声明
        this->declare_parameter("model_filename", "yolo11x_seg_bayese_640x640_nv12.bin");
        this->declare_parameter("show_image", true);
        this->declare_parameter("conf_thres", 0.50);
        this->declare_parameter("nms_thres", 0.45);
        this->declare_parameter("camera.fx", 905.5593);
        this->declare_parameter("camera.fy", 905.5208);
        this->declare_parameter("camera.cx", 663.4498);
        this->declare_parameter("camera.cy", 366.7621);

        // 使用结构体统一管理内参
        cam_param_.fx = this->get_parameter("camera.fx").as_double();
        cam_param_.fy = this->get_parameter("camera.fy").as_double();
        cam_param_.cx = this->get_parameter("camera.cx").as_double();
        cam_param_.cy = this->get_parameter("camera.cy").as_double();

        // 2. 路径解析
        std::string model_filename = this->get_parameter("model_filename").as_string();

        if (model_filename.front() == '/') {
            resolved_model_path_ = model_filename;
        } else {
            try {
                // 1. 获取包路径
                std::filesystem::path p(ament_index_cpp::get_package_share_directory("segment_yolov8_11_bpu"));

                // 2. 向上查找直到找到 'install' 目录
                while (p.has_parent_path() && p.filename() != "install") {
                    p = p.parent_path();
                }

                // 3. 取 install 的上一级作为工作空间根目录，并拼接 models
                if (p.filename() == "install") {
                    resolved_model_path_ = (p.parent_path() / "models" / model_filename).string();
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Failed to locate 'install' directory.");
                    return;
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Path error: %s", e.what());
                return;
            }
        }

        // 3. 检查与加载
        if (!std::filesystem::exists(resolved_model_path_)) {
            RCLCPP_ERROR(this->get_logger(), "[Error] Model not found: %s", resolved_model_path_.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loading model: %s", resolved_model_path_.c_str());

        // 3. Init DNN
        if (Init() != 0) {
            RCLCPP_ERROR(this->get_logger(), "Init failed!");
            return;
        }

        if (GetModelInputSize(0, model_input_w_, model_input_h_) < 0) {
             RCLCPP_ERROR(this->get_logger(), "Get model input size failed!");
        }

        segmenter_ = std::make_shared<BPU_Segment>();

        this->get_parameter("show_image", show_img_);
        double conf = this->get_parameter("conf_thres").as_double();
        double nms = this->get_parameter("nms_thres").as_double();
        
        segmenter_->config_.conf_thres = (float)conf;
        segmenter_->config_.nms_thres = (float)nms;

        callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&YoloSegNode::parameter_callback, this, std::placeholders::_1));

        // 4. QoS & Sync
        auto qos = rclcpp::SensorDataQoS().keep_last(1);
        sub_color_filter_.subscribe(this, "/camera/realsense_d435i/color/image_raw", qos.get_rmw_qos_profile());
        sub_depth_filter_.subscribe(this, "/camera/realsense_d435i/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), sub_color_filter_, sub_depth_filter_
        );
        sync_->registerCallback(std::bind(&YoloSegNode::SyncCallback, this, std::placeholders::_1, std::placeholders::_2));

        pub_ = this->create_publisher<object3d_msgs::msg::Object3DArray>("target_points_array", qos);
    }

    ~YoloSegNode() override {}

protected:
    int SetNodePara() override {
        if (!dnn_node_para_ptr_) return -1;
        dnn_node_para_ptr_->model_file = resolved_model_path_;
        dnn_node_para_ptr_->task_num = 2; 
        return 0;
    }

    int PostProcess(const std::shared_ptr<DnnNodeOutput>& node_output) override {
        if (!node_output) return -1;

        auto seg_output = std::dynamic_pointer_cast<YoloSegOutput>(node_output);
        if (!seg_output || !seg_output->src_img || !seg_output->depth_img) return -1;

        // 1. FPS 监控 (工具类)
        fps_monitor_.Tick();

        // 2. 算法处理 (返回 UnifiedResult)
        std::vector<yolo_common::UnifiedResult> results;
        segmenter_->PostProcess(node_output->output_tensors, model_input_h_, model_input_w_, results);

        // 3. 深度提取与发布 (一行代码搞定)
        if (pub_->get_subscription_count() > 0) {
            object3d_msgs::msg::Object3DArray msg;
            // 注意：BPU 代码里 seg_output->depth_img 是指针，需要解引用
            yolo_common::ros_utils::ResultsTo3DMessage(
                results, *(seg_output->depth_img), *(seg_output->msg_header), cam_param_, msg
            );
            pub_->publish(msg);
        }

        // 4. 可视化
        if (show_img_) {
            yolo_common::vis::ShowWindow("BPU Segment", *(seg_output->src_img), results, win_created_flag_, fps_monitor_.Get());
        } else if (win_created_flag_) {
            // 参数动态关闭显示时，销毁窗口
            cv::destroyWindow("BPU Segment");
            win_created_flag_ = false;
            cv::waitKey(1);
        }

        return 0;
    }

private:
    std::string resolved_model_path_;
    int model_input_w_, model_input_h_;
    bool show_img_; 
    
    // 使用公共结构体
    yolo_common::ros_utils::CameraIntrinsics cam_param_;
    // 使用公共FPS监控
    yolo_common::utils::FpsMonitor fps_monitor_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_color_filter_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_depth_filter_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr pub_;
    OnSetParametersCallbackHandle::SharedPtr callback_handle_;
    std::shared_ptr<BPU_Segment> segmenter_;

    bool win_created_flag_ = false; 

    rcl_interfaces::msg::SetParametersResult parameter_callback(
        const std::vector<rclcpp::Parameter> &params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";
        for (const auto &param : params) {
             if (param.get_name() == "conf_thres") {
                if (segmenter_) segmenter_->config_.conf_thres = (float)param.as_double();
            } else if (param.get_name() == "nms_thres") {
                if (segmenter_) segmenter_->config_.nms_thres = (float)param.as_double();
            } else if (param.get_name() == "show_image") {
                show_img_ = param.as_bool();
            }
        }
        return result;
    }

    void SyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_color, 
                      const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth) {
        // ... (保持不变，省略 try-catch 细节) ...
        cv::Mat img_color, img_depth;
        try {
            img_color = cv_bridge::toCvShare(msg_color, "bgr8")->image.clone();
            img_depth = cv_bridge::toCvShare(msg_depth, "16UC1")->image.clone();
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        cv::Mat nv12_mat;
        segmenter_->PreProcess(img_color, model_input_w_, model_input_h_, nv12_mat);

        auto pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
            reinterpret_cast<const char*>(nv12_mat.data),
            model_input_h_, model_input_w_,
            model_input_h_, model_input_w_
        );

        auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};
        auto output = std::make_shared<YoloSegOutput>();
        output->msg_header = std::make_shared<std_msgs::msg::Header>(msg_color->header);
        output->src_img = std::make_shared<cv::Mat>(img_color);
        output->depth_img = std::make_shared<cv::Mat>(img_depth);

        Run(inputs, output, nullptr, false);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloSegNode>("yolo_seg_node"));
    rclcpp::shutdown();
    return 0;
}