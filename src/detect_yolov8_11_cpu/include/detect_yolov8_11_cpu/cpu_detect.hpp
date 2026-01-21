#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

#include "yolo_common/types.hpp"

class CPU_Detect {
public:
    struct Config {
        float conf_thres = 0.50f;
        float iou_thres = 0.45f;
        std::vector<std::string> class_names;
        int class_num;
    };

    CPU_Detect(const std::string& model_path);
    ~CPU_Detect() = default;

    // 配置实例
    Config config_;

    // 核心推理函数
    std::vector<yolo_common::UnifiedResult> detect(const cv::Mat& img);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    std::string input_name_;
    std::string output_name_;
    int64_t input_w_;
    int64_t input_h_;
};