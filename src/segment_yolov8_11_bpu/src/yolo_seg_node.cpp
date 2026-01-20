#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include <chrono>
#include <fstream>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "object3d_msgs/msg/object3_d.hpp"
#include "object3d_msgs/msg/object3_d_array.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "segment_yolov8_11_bpu/bpu_seg_dnn.hpp"

// 引入 yolo_common 头文件
#include "yolo_common/visualization.hpp"
#include "yolo_common/math_utils.hpp"

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

        this->get_parameter("camera.fx", fx_);
        this->get_parameter("camera.fy", fy_);
        this->get_parameter("camera.cx", cx_);
        this->get_parameter("camera.cy", cy_);

        // 2. 路径解析
        std::string model_param = this->get_parameter("model_filename").as_string();
        if (model_param.front() == '/') {
            resolved_model_path_ = model_param;
        } else {
            try {
                resolved_model_path_ = ament_index_cpp::get_package_share_directory("segment_yolov8_11_bpu") + "/models/" + model_param;
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Error finding package path: %s", e.what());
                return;
            }
        }

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
        this->get_parameter("conf_thres", conf_thres_);
        this->get_parameter("nms_thres", nms_thres_);
        
        segmenter_->config_.conf_thres = (float)conf_thres_;
        segmenter_->config_.nms_thres = (float)nms_thres_;

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
        last_calc_time_ = std::chrono::steady_clock::now();
        last_log_time_ = std::chrono::steady_clock::now();
    }

    ~YoloSegNode() override {
        // 不需要手动 destroyAllWindows，OS 或 vis 库会处理
    }

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

        UpdateFPS(show_img_);

        // 1. 算法处理 (返回 UnifiedResult)
        std::vector<yolo_common::UnifiedResult> results;
        segmenter_->PostProcess(node_output->output_tensors, model_input_h_, model_input_w_, results);

        // 2. 深度提取与发布
        PublishSegMessage(results, seg_output->msg_header, *(seg_output->depth_img));

        // 3. 可视化 (调用 yolo_common)
        if (show_img_) {
            yolo_common::vis::ShowWindow("BPU Segment", *(seg_output->src_img), results, win_created_flag_, fps_);
        } else {
             // 如果关闭显示，需要清理窗口，yolo_common::vis 没有提供关闭接口，
             // 但 ShowWindow 内部有逻辑，如果不调用它，窗口自然不会刷新。
             // 如果需要显式关闭，可以在这里判断 win_created_flag_ 并 destroyWindow
             if (win_created_flag_) {
                cv::destroyWindow("BPU Segment");
                win_created_flag_ = false;
                cv::waitKey(1);
             }
        }

        return 0;
    }

private:
    std::string resolved_model_path_;
    int model_input_w_, model_input_h_;
    bool show_img_; 
    double conf_thres_, nms_thres_;
    double fx_, fy_, cx_, cy_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_color_filter_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_depth_filter_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr pub_;
    OnSetParametersCallbackHandle::SharedPtr callback_handle_;
    std::shared_ptr<BPU_Segment> segmenter_;

    std::chrono::steady_clock::time_point last_calc_time_, last_log_time_;
    int frame_count_ = 0;
    double fps_ = 0.0;
    bool win_created_flag_ = false; // 用于 ShowWindow

    void PublishSegMessage(const std::vector<yolo_common::UnifiedResult>& results, 
                           const std::shared_ptr<std_msgs::msg::Header>& header,
                           const cv::Mat& depth_mat) {
        
        if (pub_->get_subscription_count() == 0) return;

        object3d_msgs::msg::Object3DArray msg;
        msg.header = *header;

        for (const auto& res : results) {
            // 使用 yolo_common::math::GetRobustDepth
            // 注意：res.center 对应之前的 mic_center
            float z_m = yolo_common::math::GetRobustDepth(depth_mat, res.center);
            
            if (z_m <= 0.0f) continue; 

            float u = res.center.x;
            float v = res.center.y;
            float x_m = (u - cx_) * z_m / fx_;
            float y_m = (v - cy_) * z_m / fy_;

            float radius_m = (res.mic_radius * z_m) / fx_;
            
            object3d_msgs::msg::Object3D obj;
            obj.class_name = res.class_name;
            obj.score = res.score;
            obj.point.x = x_m;
            obj.point.y = y_m;
            obj.point.z = z_m;
            obj.width_m = radius_m * 2.0f;
            obj.height_m = radius_m * 2.0f;

            msg.objects.push_back(obj);
        }

        if (!msg.objects.empty()) {
            pub_->publish(msg);
        }
    }

    rcl_interfaces::msg::SetParametersResult parameter_callback(
        const std::vector<rclcpp::Parameter> &params) {
        // ... (参数回调逻辑保持不变)
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

    void UpdateFPS(bool show_img) {
        auto now = std::chrono::steady_clock::now();
        frame_count_++;
        auto calc_dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_calc_time_).count();
        if (calc_dur >= 1000) { 
            fps_ = frame_count_ * 1000.0 / calc_dur;
            frame_count_ = 0;
            last_calc_time_ = now;
        }
        if (!show_img) {
            auto log_dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time_).count();
            if (log_dur >= 5000) { 
                RCLCPP_INFO(this->get_logger(), "FPS: %.2f", fps_);
                last_log_time_ = now;
            }
        }
    }

    void SyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_color, 
                      const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth) {
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