#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include <chrono>
#include <fstream>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "segment/yolo_seg_common.h"
#include "segment/bpu_seg_dnn.h"

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNodeOutput;

class YoloSegNode : public hobot::dnn_node::DnnNode {
public:
    YoloSegNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : hobot::dnn_node::DnnNode(node_name, options) {
        
        // 1. 声明参数
        // 默认值只写文件名即可，程序会自动去 share/segment/models/ 下找
        this->declare_parameter("model_filename", "yolo11x_seg_bayese_640x640_nv12.bin");
        this->declare_parameter("show_image", true);
        this->declare_parameter("conf_thres", 0.50);
        this->declare_parameter("nms_thres", 0.45);
        
        // =======================================================================
        // 2. 智能路径解析 (仿 Python 逻辑)
        // =======================================================================
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

        // =======================================================================
        // 3. 初始化 DNN (Init 会调用 SetNodePara)
        // =======================================================================
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

        // QoS 设置
        rclcpp::QoS qos_profile(1); 
        qos_profile.reliability(rclcpp::ReliabilityPolicy::Reliable);
        qos_profile.history(rclcpp::HistoryPolicy::KeepLast);

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/realsense_d435i/color/image_raw", qos_profile,
            std::bind(&YoloSegNode::ColorCallback, this, std::placeholders::_1));
             
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

    int model_input_w_ = 0;
    int model_input_h_ = 0;

    bool show_img_ = true; 
    double conf_thres_ = 0.50;
    double nms_thres_ = 0.45;

    OnSetParametersCallbackHandle::SharedPtr callback_handle_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    std::shared_ptr<BPU_Segment> segmenter_;

    std::chrono::steady_clock::time_point last_calc_time_;
    std::chrono::steady_clock::time_point last_log_time_;
    int frame_count_ = 0;
    double fps_ = 0.0;

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
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloSegNode>("yolo_seg_node"));
    rclcpp::shutdown();
    return 0;
}