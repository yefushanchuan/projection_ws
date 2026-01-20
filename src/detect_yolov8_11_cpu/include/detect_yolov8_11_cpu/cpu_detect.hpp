#ifndef CPU_DETECT_HPP
#define CPU_DETECT_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 引入通用数据结构
#include "yolo_common/types.hpp"

class CPU_Detect {
public:
    CPU_Detect(const std::string& model_path, float conf_thres = 0.5, float iou_thres = 0.45);
    ~CPU_Detect() = default;

    // 核心函数：不再需要传入 show_img，只负责输出数据
    std::vector<yolo_common::UnifiedResult> detect(const cv::Mat& img);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    std::string input_name_;
    std::string output_name_;
    int64_t input_w_;
    int64_t input_h_;
    
    float conf_threshold_;
    float iou_threshold_;

    // 类别名称表
    const std::vector<std::string> class_names_;
};

#endif // CPU_DETECT_HPP