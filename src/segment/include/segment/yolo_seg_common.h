#ifndef YOLO_SEG_COMMON_H
#define YOLO_SEG_COMMON_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include "dnn_node/dnn_node_data.h"

// 检测与分割结果
struct SegResult {
    int id;
    float score;
    cv::Rect box;
    std::string class_name;
    cv::Mat mask; 
};

// 传递给 PostProcess 的自定义数据包
struct YoloOutput : public hobot::dnn_node::DnnNodeOutput {
    std::shared_ptr<cv::Mat> src_img; // 携带原始 BGR 图像用于画图
};

#endif // YOLO_SEG_COMMON_H