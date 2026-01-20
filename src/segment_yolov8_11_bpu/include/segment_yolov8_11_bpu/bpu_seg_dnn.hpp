#pragma once // 等同于 #ifndef ... #define ...

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

// 引入 BPU SDK 头文件
#include "dnn_node/dnn_node_data.h"
#include "std_msgs/msg/header.hpp"

// 引入统一数据结构
#include "yolo_common/types.hpp"

// ==========================================
// 1. 定义数据结构 (原 yolo_seg_common.h 内容)
// ==========================================

// 传递给 PostProcess 的自定义数据包
// 这是 dnn_node 框架要求的，用于在 Tensor 和 原始图像之间传参
struct YoloSegOutput : public hobot::dnn_node::DnnNodeOutput {
    std::shared_ptr<std_msgs::msg::Header> msg_header;
    std::shared_ptr<cv::Mat> src_img;
    std::shared_ptr<cv::Mat> depth_img;
};

// ==========================================
// 2. 定义推理类 (原 bpu_seg_dnn.h 内容)
// ==========================================

class BPU_Segment {
public:
    struct Config {
        int class_num = 80;
        int reg_max = 16;
        int num_mask = 32;
        std::vector<int> strides = {8, 16, 32};
        float conf_thres = 0.50;
        float nms_thres = 0.45;
        std::vector<std::string> class_names;
    };

    BPU_Segment();
    ~BPU_Segment() = default;

    Config config_;

    // 预处理：BGR -> NV12 (BPU要求)
    void PreProcess(const cv::Mat& bgr_img, int model_w, int model_h, cv::Mat& nv12_out);

    // 后处理：Tensor -> UnifiedResult
    void PostProcess(const std::vector<std::shared_ptr<hobot::dnn_node::DNNTensor>>& tensors,
                     int model_h, int model_w,
                     std::vector<yolo_common::UnifiedResult>& results);

private:
    // 用于坐标还原的参数
    float ratio_ = 1.0f;
    int pad_w_ = 0;
    int pad_h_ = 0;

    // 内部辅助函数
    void Letterbox(const cv::Mat& src, int target_w, int target_h, cv::Mat& dst);
    void BGRToNV12(const cv::Mat& bgr, cv::Mat& nv12);
};