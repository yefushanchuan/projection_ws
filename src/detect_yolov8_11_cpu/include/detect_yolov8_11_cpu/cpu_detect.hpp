#ifndef CPU_DETECT_HPP
#define CPU_DETECT_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 检测结果结构体
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box; // x, y, w, h
    cv::Point center;
};

class CPU_Detect {
public:
    // 构造函数
    CPU_Detect(const std::string& model_path, float conf_thres = 0.5, float iou_thres = 0.45);
    
    // 析构函数
    ~CPU_Detect() = default;

    // 推理核心函数
    std::vector<Detection> detect(cv::Mat& img, bool show_img);

    // 获取类别名称的辅助函数
    std::string getClassName(int id) const;

private:
    // ONNX Runtime 成员变量
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    // 模型信息
    std::string input_name_;
    std::string output_name_;
    int64_t input_w_;
    int64_t input_h_;
    
    // 参数
    float conf_threshold_;
    float iou_threshold_;
};

#endif