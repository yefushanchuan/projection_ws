#pragma once
#include <opencv2/opencv.hpp>

namespace yolo_common {
namespace proc {

    /**
     * @brief 保持长宽比缩放并填充灰边 (Letterbox)
     * @param src 输入图像
     * @param dst 输出图像
     * @param target_w 目标宽
     * @param target_h 目标高
     * @param pad_w [输出] 宽度方向的填充量
     * @param pad_h [输出] 高度方向的填充量
     * @return float 缩放比例 (scale)
     */
    inline float Letterbox(const cv::Mat& src, cv::Mat& dst, int target_w, int target_h, 
                           int& pad_w, int& pad_h) {
        int in_w = src.cols;
        int in_h = src.rows;
        float scale = std::min((float)target_w / in_w, (float)target_h / in_h);
        
        int new_w = std::round(in_w * scale);
        int new_h = std::round(in_h * scale);
        
        pad_w = (target_w - new_w) / 2;
        pad_h = (target_h - new_h) / 2;

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(new_w, new_h));
        
        // 使用 114 填充灰边 (YOLO 标准)
        cv::copyMakeBorder(resized, dst, pad_h, target_h - new_h - pad_h, 
                           pad_w, target_w - new_w - pad_w, 
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        
        return scale;
    }

} // namespace proc
} // namespace yolo_common