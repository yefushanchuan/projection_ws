#pragma once
#include <string>
#include <opencv2/core.hpp>

namespace yolo_common {

struct UnifiedResult {
    int id;                     // 类别ID
    float score;                // 置信度
    std::string class_name;     // 类别名称
    
    cv::Rect box;               // 2D 边界框 (xywh)
    cv::Mat mask;               // 分割掩码 (如果是纯检测任务，此项为空)
    
    cv::Point2f center;         // 用于提取深度的中心点
                                // (检测用 box 中心，分割可用 mask 重心)
                                
    float mic_radius = 0.0f;    // 分割专用：最大内切圆半径 (检测任务忽略)
};

} // namespace yolo_common