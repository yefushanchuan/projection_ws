#ifndef BPU_SEG_DNN_H
#define BPU_SEG_DNN_H

#include "yolo_seg_common.h"
#include <vector>
#include <string>

// 定义配置结构体
struct Config {
    int class_num = 80;
    int reg_max = 16;
    int num_mask = 32;
    std::vector<int> strides = {8, 16, 32};
    float conf_thres = 0.50;
    float nms_thres = 0.45;
    std::vector<std::string> class_names;
};

class BPU_Segment {
public:
    BPU_Segment();
    ~BPU_Segment();

    Config config_;

    // 1. 预处理
    void PreProcess(const cv::Mat& bgr_img, int model_w, int model_h, cv::Mat& nv12_out);

    // 2. 后处理
    void PostProcess(const std::vector<std::shared_ptr<hobot::dnn_node::DNNTensor>>& tensors,
                     int model_h, int model_w,
                     std::vector<SegResult>& results);

    // 3. 可视化绘制 (整合了 draw 和 show)
    void detect_result(cv::Mat& img, const std::vector<SegResult>& results, double fps, bool show_img);

private:
    bool window_created_ = false;

    // === 关键：用于坐标还原的成员变量 ===
    float ratio_ = 1.0f;
    int pad_w_ = 0;
    int pad_h_ = 0;

    // === 关键：内部辅助函数声明 ===
    void Letterbox(const cv::Mat& src, int target_w, int target_h, cv::Mat& dst);
    void BGRToNV12(const cv::Mat& bgr, cv::Mat& nv12);
    void draw_detection(cv::Mat& img, const SegResult& res);
    cv::Scalar GetColor(int id);
};

#endif // BPU_SEG_DNN_H