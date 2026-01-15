#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include <chrono>

// 引入分层后的头文件
#include "segment/yolo_seg_common.h"
#include "segment/bpu_seg_hobot_dnn.h"

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNodeOutput;

class YoloSegNode : public hobot::dnn_node::DnnNode {
public:
    YoloSegNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : hobot::dnn_node::DnnNode(node_name, options) {
        
        // 1. 声明参数
        this->declare_parameter("model_filename", "/home/sunrise/projection_ws/src/segment/models/yolo11x_seg_bayese_640x640_nv12.bin");
        this->declare_parameter("show_image", true); // 仿照 python 增加开关

        // 2. 初始化 DNN
        if (Init() != 0) {
            RCLCPP_ERROR(this->get_logger(), "Init failed!");
            return;
        }

        // 3. 获取模型输入尺寸
        if (GetModelInputSize(0, model_input_w_, model_input_h_) < 0) {
             RCLCPP_ERROR(this->get_logger(), "Get model input size failed!");
        }

        // 4. 初始化算法引擎 (相当于 Python 的 self.detector = BPU_Detect(...))
        segmenter_ = std::make_shared<yolo_seg::BPU_Segment>();

        // 5. 订阅与发布
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/realsense_d435i/color/image_raw", 10,
            std::bind(&YoloSegNode::ColorCallback, this, std::placeholders::_1));
            
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("yolo_seg_visual", 10);
        
        // FPS 统计
        last_calc_time_ = std::chrono::steady_clock::now();
        last_log_time_ = std::chrono::steady_clock::now();
    }

protected:
    // 设置 DNN 参数
    int SetNodePara() override {
        if (!dnn_node_para_ptr_) return -1;
        dnn_node_para_ptr_->model_file = this->get_parameter("model_filename").as_string();
        dnn_node_para_ptr_->task_num = 2; 
        return 0;
    }

    // 推理完成后的回调 (Process)
    int PostProcess(const std::shared_ptr<DnnNodeOutput>& node_output) override {
        if (!node_output) return -1;

        // 1. 获取上下文 (原图)
        auto yolo_output = std::dynamic_pointer_cast<yolo_seg::YoloOutput>(node_output);
        if (!yolo_output || !yolo_output->src_img) return -1;

        // 2. 计算 FPS
        UpdateFPS();

        // 3. 调用引擎进行后处理 (PostProcess)
        std::vector<yolo_seg::SegResult> results;
        segmenter_->PostProcess(node_output->output_tensors, model_input_h_, model_input_w_, results);
        
        // 4. 调用引擎进行可视化 (Visualize)
        // 获取参数开关，类似于 Python 中的 detect_result(show_img=...)
        bool show_img = this->get_parameter("show_image").as_bool();
        
        // 准备画图用的 Mat (深拷贝)
        cv::Mat draw_img = yolo_output->src_img->clone();
        
        segmenter_->Visualize(draw_img, results, model_input_w_, model_input_h_, fps_, show_img);

        // 5. 发布 ROS 话题
        if (pub_->get_subscription_count() > 0) {
            std_msgs::msg::Header header = *(node_output->msg_header);
            auto msg = cv_bridge::CvImage(header, "bgr8", draw_img).toImageMsg();
            pub_->publish(*msg);
        }

        return 0;
    }

private:
    int model_input_w_ = 0;
    int model_input_h_ = 0;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    
    // 算法引擎实例
    std::shared_ptr<yolo_seg::BPU_Segment> segmenter_;

    // FPS 变量
    std::chrono::steady_clock::time_point last_calc_time_;
    std::chrono::steady_clock::time_point last_log_time_;
    int frame_count_ = 0;
    double fps_ = 0.0;

    void UpdateFPS() {
        auto now = std::chrono::steady_clock::now();
        frame_count_++;
        auto calc_dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_calc_time_).count();
        if (calc_dur >= 1000) { 
            fps_ = frame_count_ * 1000.0 / calc_dur;
            frame_count_ = 0;
            last_calc_time_ = now;
        }
        auto log_dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time_).count();
        if (log_dur >= 5000) {
            RCLCPP_INFO(this->get_logger(), "Current FPS: %.2f", fps_);
            last_log_time_ = now;
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

        // 1. 统一缩放至 1280x720 (保持原逻辑)
        cv::Mat display_img;
        if (cv_ptr->image.cols != 1280 || cv_ptr->image.rows != 720) {
            cv::resize(cv_ptr->image, display_img, cv::Size(1280, 720));
        } else {
            display_img = cv_ptr->image;
        }

        // 2. 调用引擎进行预处理 (PreProcess -> NV12)
        cv::Mat nv12_mat;
        segmenter_->PreProcess(display_img, model_input_w_, model_input_h_, nv12_mat);

        // 3. 构建 BPU 输入
        auto pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
            reinterpret_cast<const char*>(nv12_mat.data),
            model_input_h_, model_input_w_, 
            model_input_h_, model_input_w_
        );
        auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};

        // 4. 构建上下文输出 (携带原图)
        auto output = std::make_shared<yolo_seg::YoloOutput>();
        output->msg_header = std::make_shared<std_msgs::msg::Header>(msg->header);
        output->src_img = std::make_shared<cv::Mat>(display_img.clone()); 

        // 5. 执行推理
        Run(inputs, output, nullptr, false);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloSegNode>("yolo_seg_node"));
    rclcpp::shutdown();
    return 0;
}