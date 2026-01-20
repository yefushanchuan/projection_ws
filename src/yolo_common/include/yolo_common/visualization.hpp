#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "types.hpp"

namespace yolo_common {
namespace vis {

inline cv::Scalar GetColor(int id) {
    int r = (id * 123 + 45) % 255;
    int g = (id * 234 + 90) % 255;
    int b = (id * 345 + 135) % 255;
    return cv::Scalar(b, g, r);
}

// 绘制单个结果 (支持 Box 和 Mask)
inline void DrawResult(cv::Mat& img, const UnifiedResult& res) {
    cv::Scalar color = GetColor(res.id);

    // 1. 绘制 Mask (如果有)
    if (!res.mask.empty()) {
        cv::Rect valid_box = res.box & cv::Rect(0, 0, img.cols, img.rows);
        if (valid_box.area() > 0) {
            cv::Mat mask_roi;
            // 假设 res.mask 已经是裁剪好的 ROI 或需要 resize，这里简化处理，假设已 resize 到 box 大小
            if (res.mask.size() != res.box.size()) {
                cv::resize(res.mask, mask_roi, res.box.size());
            } else {
                mask_roi = res.mask;
            }
            
            // 处理边界裁剪
            int x_offset = valid_box.x - res.box.x;
            int y_offset = valid_box.y - res.box.y;
            cv::Rect mask_crop_rect(x_offset, y_offset, valid_box.width, valid_box.height);
            mask_roi = mask_roi(mask_crop_rect);

            cv::Mat mask_bin = mask_roi > 0.5;
            
            // 颜色混合
            cv::Mat img_roi = img(valid_box);
            cv::Mat color_layer(img_roi.size(), CV_8UC3, color);
            cv::Mat temp;
            cv::addWeighted(img_roi, 0.6, color_layer, 0.4, 0.0, temp);
            temp.copyTo(img_roi, mask_bin);
        }
    }

    // 2. 绘制 Box
    cv::rectangle(img, res.box, color, 2);

    // 3. 绘制标签
    std::string label = res.class_name + " " + std::to_string((int)(res.score * 100)) + "%";
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int label_y = std::max(res.box.y, label_size.height + 10);
    
    cv::rectangle(img, cv::Point(res.box.x, label_y - label_size.height - 5), 
                  cv::Point(res.box.x + label_size.width, label_y + 5), color, -1);
    cv::putText(img, label, cv::Point(res.box.x, label_y), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

// --- 核心：窗口管理与显示 (含复活逻辑) ---
inline void ShowWindow(const std::string& win_name, const cv::Mat& img, 
                       const std::vector<UnifiedResult>& results, 
                       bool& win_created_flag, 
                       double fps = 0.0) 
{
    // 1. 绘制所有结果
    cv::Mat draw_img = img.clone(); // 避免修改原图
    for (const auto& res : results) {
        DrawResult(draw_img, res);
    }
    
    if (fps > 0) {
        cv::putText(draw_img, "FPS: " + cv::format("FPS: %.2f", fps), 
                    cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    // 2. 窗口状态检查 (自动复活逻辑)
    try {
        if (win_created_flag && cv::getWindowProperty(win_name, cv::WND_PROP_VISIBLE) < 1.0) {
            win_created_flag = false; // 窗口被用户关闭，重置标志
        }
    } catch (...) {
        win_created_flag = false;
    }

    // 3. 创建窗口
    if (!win_created_flag) {
        try {
            cv::namedWindow(win_name, cv::WINDOW_NORMAL);
            cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            win_created_flag = true;
        } catch(...) { return; }
    }

    // 4. 显示
    if (win_created_flag) {
        cv::imshow(win_name, draw_img);
        cv::waitKey(1);
    }
}

} // namespace vis
} // namespace yolo_common