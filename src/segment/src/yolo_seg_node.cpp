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

#include "segment/yolo_seg_common.h"
#include "segment/bpu_seg_dnn.h"

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNodeOutput;

class YoloSegNode : public hobot::dnn_node::DnnNode {
public:
    YoloSegNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : hobot::dnn_node::DnnNode(node_name, options) {
        // 1. 参数声明与获取
        // 默认值只写文件名即可，程序会自动去 share/segment/models/ 下找
        this->declare_parameter("model_filename", "yolo11x_seg_bayese_640x640_nv12.bin");
        this->declare_parameter("show_image", true);
        this->declare_parameter("conf_thres", 0.50);
        this->declare_parameter("nms_thres", 0.45);

        // 相机内参 (默认值来自你的 Detect 代码)
        this->declare_parameter("camera.fx", 905.5593);
        this->declare_parameter("camera.fy", 905.5208);
        this->declare_parameter("camera.cx", 663.4498);
        this->declare_parameter("camera.cy", 366.7621);

        this->get_parameter("camera.fx", fx_);
        this->get_parameter("camera.fy", fy_);
        this->get_parameter("camera.cx", cx_);
        this->get_parameter("camera.cy", cy_);

        // 2. 智能路径解析 (仿 Python 逻辑)
        std::string model_param = this->get_parameter("model_filename").as_string();
        
        // 判断是否为绝对路径 (以 / 开头)
        if (model_param.front() == '/') {
            resolved_model_path_ = model_param;
        } else {
            // 如果是相对路径/文件名，则拼接 share 目录
            try {
                resolved_model_path_ = ament_index_cpp::get_package_share_directory("segment") + "/models/" + model_param;
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Can not find package 'segment': %s", e.what());
                return;
            }
        }

        // 检查文件是否存在
        std::ifstream f(resolved_model_path_.c_str());
        if (!f.good()) {
            RCLCPP_ERROR(this->get_logger(), "Model file not found: %s", resolved_model_path_.c_str());
            // 这里不 return，让 Init() 去报错，或者你可以选择直接 throw
        } else {
            RCLCPP_INFO(this->get_logger(), "Loading Model: %s", resolved_model_path_.c_str());
        }

        // 3. 初始化 DNN (Init 会调用 SetNodePara)
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
        
        // 将成员变量的值同步给算法引擎
        segmenter_->config_.conf_thres = (float)conf_thres_;
        segmenter_->config_.nms_thres = (float)nms_thres_;

        callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&YoloSegNode::parameter_callback, this, std::placeholders::_1));

        // 4. QoS 设置
        rclcpp::QoS qos_profile(1); 
        qos_profile.reliability(rclcpp::ReliabilityPolicy::Reliable);
        qos_profile.history(rclcpp::HistoryPolicy::KeepLast);

        sub_color_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/realsense_d435i/color/image_raw", qos_profile,
            std::bind(&YoloSegNode::ColorCallback, this, std::placeholders::_1));
        
        sub_depth_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/realsense_d435i/aligned_depth_to_color/image_raw", qos_profile,
            std::bind(&YoloSegNode::DepthCallback, this, std::placeholders::_1));

        pub_ = this->create_publisher<object3d_msgs::msg::Object3DArray>("target_points_array", qos_profile);

        last_calc_time_ = std::chrono::steady_clock::now();
        last_log_time_ = std::chrono::steady_clock::now();
    }

protected:
    int SetNodePara() override {
        if (!dnn_node_para_ptr_) return -1;
        // 使用我们在构造函数中解析好的绝对路径
        dnn_node_para_ptr_->model_file = resolved_model_path_;
        dnn_node_para_ptr_->task_num = 2; 
        return 0;
    }

    int PostProcess(const std::shared_ptr<DnnNodeOutput>& node_output) override {
        if (!node_output) return -1;

        auto yolo_output = std::dynamic_pointer_cast<YoloOutput>(node_output);
        if (!yolo_output || !yolo_output->src_img) return -1;

        UpdateFPS(show_img_);

        // 算法后处理
        std::vector<SegResult> results;
        segmenter_->PostProcess(node_output->output_tensors, model_input_h_, model_input_w_, results);

        PublishSegMessage(results, yolo_output->msg_header);

        // 可视化
        if (show_img_) {
            cv::Mat draw_img = yolo_output->src_img->clone();
            segmenter_->detect_result(draw_img, results, fps_, true);
        } else {
            cv::Mat dummy_img; 
            segmenter_->detect_result(dummy_img, results, fps_, false);
        }

        return 0;
    }

private:
    std::string resolved_model_path_; // 存储解析后的模型路径

    int model_input_w_;
    int model_input_h_;

    bool show_img_; 
    double conf_thres_;
    double nms_thres_;
    double fx_, fy_, cx_, cy_;

    OnSetParametersCallbackHandle::SharedPtr callback_handle_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_color_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr pub_;

    std::shared_ptr<BPU_Segment> segmenter_;

    cv::Mat depth_image_;
    bool depth_ready_ = false;

    std::chrono::steady_clock::time_point last_calc_time_;
    std::chrono::steady_clock::time_point last_log_time_;
    int frame_count_ = 0;
    double fps_ = 0.0;

    void PublishSegMessage(const std::vector<SegResult>& results, 
                           const std::shared_ptr<std_msgs::msg::Header>& header) {
        // 1. 检查有没有人订阅，省流
        if (pub_->get_subscription_count() == 0) return;
        
        // 2. 检查深度图是否就绪
        if (!depth_ready_ || depth_image_.empty()) {
            // 可以选择打印一次 debug，避免刷屏
            // RCLCPP_DEBUG(this->get_logger(), "Depth image not ready yet.");
            return;
        }

        object3d_msgs::msg::Object3DArray msg;
        msg.header = *header; // 保持时间戳同步

        for (const auto& res : results) {
            // 获取内切圆中心坐标 (像素)
            int u = std::round(res.mic_center.x);
            int v = std::round(res.mic_center.y);

            // 越界检查
            if (u < 0 || u >= depth_image_.cols || v < 0 || v >= depth_image_.rows) continue;

            // 获取深度值 (mm -> m)
            // Realsense 默认是 16UC1，单位毫米
            uint16_t depth_mm = depth_image_.at<uint16_t>(v, u);
            if (depth_mm == 0) continue; // 无效深度

            float z_m = static_cast<float>(depth_mm) / 1000.0f;

            // 3D 坐标转换 (像素坐标系 -> 相机坐标系)
            // X = (u - cx) * Z / fx
            // Y = (v - cy) * Z / fy
            float x_m = (static_cast<float>(u) - cx_) * z_m / fx_;
            float y_m = (static_cast<float>(v) - cy_) * z_m / fy_;

            // === 核心逻辑优化：半径像素 -> 物理半径 ===
            // 物理长度 = (像素长度 * Z) / fx
            // 这里将 mic_radius 作为像素长度，映射到 width_m 和 height_m
            // 假设物体是圆球，宽和高都是直径 (2 * radius)
            // 但你的要求是：width_m, height_m 变为 "最大内切圆半径" (物理单位)
            
            float radius_m = (res.mic_radius * z_m) / fx_;

            object3d_msgs::msg::Object3D obj;
            obj.class_name = res.class_name;
            obj.score = res.score;
            
            obj.point.x = x_m;
            obj.point.y = y_m;
            obj.point.z = z_m;

            // 按照你的要求：传递物理单位的内切圆半径
            obj.width_m = radius_m;
            obj.height_m = radius_m;

            msg.objects.push_back(obj);
        }

        if (!msg.objects.empty()) {
            pub_->publish(msg);
        }
    }

    rcl_interfaces::msg::SetParametersResult parameter_callback(
        const std::vector<rclcpp::Parameter> &params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";

        for (const auto &param : params) {
            if (param.get_name() == "conf_thres") {
                if (segmenter_) {
                    segmenter_->config_.conf_thres = (float)param.as_double();
                    RCLCPP_INFO(this->get_logger(), "Updated conf_thres: %.2f", segmenter_->config_.conf_thres);
                }
            }
            else if (param.get_name() == "nms_thres") {
                if (segmenter_) {
                    segmenter_->config_.nms_thres = (float)param.as_double();
                    RCLCPP_INFO(this->get_logger(), "Updated nms_thres: %.2f", segmenter_->config_.nms_thres);
                }
            }
            else if (param.get_name() == "show_image") {
                show_img_ = param.as_bool();
                RCLCPP_INFO(this->get_logger(), "Updated show_image: %s", show_img_ ? "true" : "false");
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
                RCLCPP_INFO(this->get_logger(), "Node Running - FPS: %.2f", fps_);
                last_log_time_ = now;
            }
        }
    }

    void ColorCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!rclcpp::ok()) return;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat display_img;
        if (cv_ptr->image.cols != 1280 || cv_ptr->image.rows != 720) {
            cv::resize(cv_ptr->image, display_img, cv::Size(1280, 720));
        } else {
            display_img = cv_ptr->image;
        }

        cv::Mat nv12_mat;
        segmenter_->PreProcess(display_img, model_input_w_, model_input_h_, nv12_mat);

        auto pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
            reinterpret_cast<const char*>(nv12_mat.data),
            model_input_h_, model_input_w_, 
            model_input_h_, model_input_w_
        );
        auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};

        auto output = std::make_shared<YoloOutput>();
        output->msg_header = std::make_shared<std_msgs::msg::Header>(msg->header);
        output->src_img = std::make_shared<cv::Mat>(display_img.clone()); 

        Run(inputs, output, nullptr, false);
    }

    void DepthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // 使用 cv_bridge 转换，zero-copy 可能更高效，这里用 toCvCopy 保证安全
            // TYPE_16UC1 是 Realsense 标准格式
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            depth_image_ = cv_ptr->image.clone();
            depth_ready_ = true;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Depth cv_bridge error: %s", e.what());
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloSegNode>("yolo_seg_node"));
    rclcpp::shutdown();
    return 0;
}