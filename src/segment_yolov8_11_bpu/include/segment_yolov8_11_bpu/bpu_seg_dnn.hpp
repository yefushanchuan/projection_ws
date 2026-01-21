#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "dnn_node/dnn_node_data.h"
#include "std_msgs/msg/header.hpp"

#include "yolo_common/types.hpp"

struct YoloSegOutput : public hobot::dnn_node::DnnNodeOutput {
    std::shared_ptr<std_msgs::msg::Header> msg_header;
    std::shared_ptr<cv::Mat> src_img;
    std::shared_ptr<cv::Mat> depth_img;
};

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
    void BGRToNV12(const cv::Mat& bgr, cv::Mat& nv12);
};