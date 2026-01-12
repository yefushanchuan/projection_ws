#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <opencv2/highgui/highgui.hpp>

// 引入你的分割解析器头文件
#include "segment/yolo_seg_node.h"

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNodeOutput;

// 自定义输出结构，携带原图
struct YoloOutput : public DnnNodeOutput {
    std::shared_ptr<cv::Mat> src_img; 
};

class YoloSegNode : public hobot::dnn_node::DnnNode {
public:
    YoloSegNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : hobot::dnn_node::DnnNode(node_name, options) {
        
        // =======================================================================
        // 1. 先声明参数 (必须放在 Init() 之前 ！！！)
        // =======================================================================
        this->declare_parameter("model_filename", "/home/sunrise/projection_ws/src/segment/models/yolov8n_seg_bayese_640x640_nv12.bin");

        // =======================================================================
        // 2. 再初始化 DNN (Init 会调用 SetNodePara 读取上面的参数)
        // =======================================================================
        if (Init() != 0) {
            RCLCPP_ERROR(this->get_logger(), "Init failed!");
            return;
        }

        // 3. 获取模型输入尺寸
        if (GetModelInputSize(0, model_input_w_, model_input_h_) < 0) {
             RCLCPP_ERROR(this->get_logger(), "Get model input size failed!");
        }

        // 4. 订阅 RGB 图像
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 10,
            std::bind(&YoloSegNode::ColorCallback, this, std::placeholders::_1));
            
        // 5. 发布可视化结果
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("yolo_seg_visual", 10);
        
        parser_ = std::make_shared<yolo_seg::Parser>();
        
        // FPS 计算初始化
        auto now = std::chrono::steady_clock::now();
        last_calc_time_ = now;
        last_log_time_ = now;
        frame_count_ = 0;
        fps_ = 0.0;
    }

protected:
    int SetNodePara() override {
        if (!dnn_node_para_ptr_) return -1;
        std::string model_file = this->get_parameter("model_filename").as_string();
        dnn_node_para_ptr_->model_file = model_file;
        dnn_node_para_ptr_->task_num = 2; 
        return 0;
    }

    int PostProcess(const std::shared_ptr<DnnNodeOutput>& node_output) override {
        if (!node_output) return -1;

        // 1. 获取原图
        auto yolo_output = std::dynamic_pointer_cast<YoloOutput>(node_output);
        if (!yolo_output || !yolo_output->src_img) return -1;

        // 2. 解析模型输出
        std::vector<yolo_seg::SegResult> results;
        parser_->Parse(node_output->output_tensors, model_input_h_, model_input_w_, results);
        
        // 3. 计算 FPS
        UpdateFPS();

        // 4. 绘制结果
        // 准备一张用于画图的 Mat (从原图深拷贝)
        cv::Mat draw_img = yolo_output->src_img->clone();
        
        // 计算缩放比例
        float x_scale = (float)draw_img.cols / model_input_w_;
        float y_scale = (float)draw_img.rows / model_input_h_;

        for (const auto& res : results) {
            // A. 颜色
            cv::Scalar color = GetColor(res.id);

            // B. 处理 Mask 
            cv::Mat mask_resized;
            if (res.mask.size() != draw_img.size()) {
                cv::resize(res.mask, mask_resized, draw_img.size(), 0, 0, cv::INTER_NEAREST);
            } else {
                mask_resized = res.mask;
            }

            // C. 绘制半透明 Mask
            cv::Mat color_mask(draw_img.size(), CV_8UC3, color);
            cv::Mat roi;
            draw_img.copyTo(roi, mask_resized);
            cv::addWeighted(roi, 0.6, color_mask, 0.4, 0.0, roi);
            roi.copyTo(draw_img, mask_resized);

            // D. 绘制检测框
            cv::Rect rect_scaled;
            rect_scaled.x = res.box.x * x_scale;
            rect_scaled.y = res.box.y * y_scale;
            rect_scaled.width = res.box.width * x_scale;
            rect_scaled.height = res.box.height * y_scale;
            
            cv::rectangle(draw_img, rect_scaled, color, 2);

            // E. 绘制标签
            std::string label = res.class_name + " " + std::to_string((int)(res.score * 100)) + "%";
            cv::putText(draw_img, label, cv::Point(rect_scaled.x, rect_scaled.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        // F. 绘制 FPS
        std::string fps_text = "FPS: " + std::to_string((int)fps_);
        cv::putText(draw_img, fps_text, cv::Point(20, 50), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);

        // ==========================================================
        // !!! 全屏显示逻辑 !!!
        // ==========================================================
        std::string win_name = "YOLOv11 Seg Visualization";

        // 只在第一次运行时设置全屏属性，避免每帧重复设置导致闪烁
        if (!window_created_) {
            cv::namedWindow(win_name, cv::WINDOW_NORMAL); // 允许自由调整大小
            cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN); // 设置全屏属性
            window_created_ = true;
        }

        cv::imshow(win_name, draw_img);
        cv::waitKey(1);
        // ==========================================================

        // 5. 发布图像 (保留这个，方便远程调试)
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
    bool window_created_ = false; 
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::shared_ptr<yolo_seg::Parser> parser_;

    // FPS 统计变量
    std::chrono::steady_clock::time_point last_calc_time_; // 上次计算FPS的时间
    std::chrono::steady_clock::time_point last_log_time_;  // 上次打印日志的时间
    int frame_count_;
    double fps_;

    void UpdateFPS() {
        auto now = std::chrono::steady_clock::now();
        frame_count_++;

        // 逻辑1: 计算 FPS (用于 OSD 显示) - 周期 1秒 (1000ms)
        auto calc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_calc_time_).count();
        if (calc_duration >= 1000) { 
            fps_ = frame_count_ * 1000.0 / calc_duration;
            frame_count_ = 0;
            last_calc_time_ = now;
        }

        // 逻辑2: 打印 FPS (用于终端 Log) - 周期 5秒 (5000ms)
        auto log_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time_).count();
        if (log_duration >= 5000) {
            // 使用上面计算好的 fps_ 值
            RCLCPP_INFO(this->get_logger(), "Current FPS: %.2f", fps_);
            last_log_time_ = now;
        }
    }

    cv::Scalar GetColor(int id) {
        int r = (id * 123 + 45) % 255;
        int g = (id * 234 + 90) % 255;
        int b = (id * 345 + 135) % 255;
        return cv::Scalar(b, g, r);
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

        // 1. 强制 Resize 到 1280x720 (作为显示用的图)
        // 无论输入是什么分辨率，这里都强制转为 1280x720
        cv::Mat display_img;
        if (cv_ptr->image.cols != 1280 || cv_ptr->image.rows != 720) {
            cv::resize(cv_ptr->image, display_img, cv::Size(1280, 720));
        } else {
            display_img = cv_ptr->image;
        }

        // 2. Resize 到模型输入大小 (640x640) 用于推理
        cv::Mat model_input_img;
        cv::resize(display_img, model_input_img, cv::Size(model_input_w_, model_input_h_));
        
        // 3. 转 NV12
        cv::Mat nv12_mat;
        bgr_to_nv12(model_input_img, nv12_mat);

        // 4. 构建输入
        auto pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
            reinterpret_cast<const char*>(nv12_mat.data),
            model_input_h_, model_input_w_, 
            model_input_h_, model_input_w_
        );
        auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};

        // 5. 构建输出 (携带 1280x720 的 display_img)
        auto output = std::make_shared<YoloOutput>();
        output->msg_header = std::make_shared<std_msgs::msg::Header>(msg->header);
        // 重要：这里存入的是 clone 的副本，防止后续被修改或内存问题
        output->src_img = std::make_shared<cv::Mat>(display_img.clone()); 

        Run(inputs, output, nullptr, false);
    }

    void bgr_to_nv12(const cv::Mat& bgr, cv::Mat& nv12) {
        int w = bgr.cols;
        int h = bgr.rows;
        cv::Mat yuv_i420;
        cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);
        nv12.create(h * 1.5, w, CV_8UC1);
        memcpy(nv12.data, yuv_i420.data, w * h);
        uint8_t* u_ptr = yuv_i420.data + w * h;
        uint8_t* v_ptr = u_ptr + (w * h) / 4;
        uint8_t* uv_ptr = nv12.data + w * h;
        for (int i = 0; i < (w * h) / 4; ++i) {
            *uv_ptr++ = *u_ptr++;
            *uv_ptr++ = *v_ptr++;
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloSegNode>("yolo_seg_node"));
    rclcpp::shutdown();
    return 0;
}