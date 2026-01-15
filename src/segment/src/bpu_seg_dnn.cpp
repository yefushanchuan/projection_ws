#include "segment/bpu_seg_dnn.h"
#include <cmath>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>

// 辅助数学函数
inline float fastExp(float x) {
    union { uint32_t i; float f; } v;
    v.i = (12102203.1616540672f * x + 1064807160.56887296f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

BPU_Segment::BPU_Segment() {
    config_.class_names.resize(80, "object");
    // 填充部分类别名用于测试
    const char* names[] = { 
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light" 
    }; 
    for(int i=0; i<10; ++i) config_.class_names[i] = names[i];
}

void BPU_Segment::BGRToNV12(const cv::Mat& bgr, cv::Mat& nv12) {
    int w = bgr.cols;
    int h = bgr.rows;
    cv::Mat yuv_i420;
    cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);
    nv12.create(h * 1.5, w, CV_8UC1);
    memcpy(nv12.data, yuv_i420.data, w * h);
    uint8_t* u_ptr = yuv_i420.data + w * h;
    uint8_t* v_ptr = u_ptr + (w * h) / 4;
    uint8_t* uv_ptr = nv12.data + w * h;
    for (int i = 0; i < (w * h) / 4; ++i) {
        *uv_ptr++ = *u_ptr++;
        *uv_ptr++ = *v_ptr++;
    }
}

// 核心：Letterbox 预处理
void BPU_Segment::Letterbox(const cv::Mat& src, int target_w, int target_h, cv::Mat& dst) {
    int in_w = src.cols;
    int in_h = src.rows;
    float scale = std::min((float)target_w / in_w, (float)target_h / in_h);
    
    int new_w = std::round(in_w * scale);
    int new_h = std::round(in_h * scale);
    
    // 保存参数供 PostProcess 使用
    ratio_ = scale;
    pad_w_ = (target_w - new_w) / 2;
    pad_h_ = (target_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    
    // 填充灰色 (114)
    cv::copyMakeBorder(resized, dst, pad_h_, target_h - new_h - pad_h_, 
                       pad_w_, target_w - new_w - pad_w_, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}

void BPU_Segment::PreProcess(const cv::Mat& bgr_img, int model_w, int model_h, cv::Mat& nv12_out) {
    // 1. 使用 Letterbox 而不是直接 Resize
    cv::Mat letterboxed;
    Letterbox(bgr_img, model_w, model_h, letterboxed);
    
    // 2. 转 NV12
    BGRToNV12(letterboxed, nv12_out);
}

void BPU_Segment::PostProcess(const std::vector<std::shared_ptr<hobot::dnn_node::DNNTensor>>& tensors,
                             int model_h, int model_w,
                             std::vector<SegResult>& results) {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> mask_coeffs;

    // 1. 解析 Head
    for (int i = 0; i < 3; ++i) {
        if ((size_t)(i * 3 + 2) >= tensors.size()) break;
        int stride = config_.strides[i];
        
        auto& cls_tensor = tensors[i * 3];
        auto& box_tensor = tensors[i * 3 + 1];
        auto& msk_tensor = tensors[i * 3 + 2];
        
        float* cls_data = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
        float* box_data = reinterpret_cast<float*>(box_tensor->sysMem[0].virAddr);
        float* msk_data = reinterpret_cast<float*>(msk_tensor->sysMem[0].virAddr);

        int h_dim = cls_tensor->properties.validShape.dimensionSize[1]; 
        int w_dim = cls_tensor->properties.validShape.dimensionSize[2]; 

        int cls_step = config_.class_num;
        int box_step = 4 * config_.reg_max;
        int msk_step = config_.num_mask;

        for (int y = 0; y < h_dim; ++y) {
            for (int x = 0; x < w_dim; ++x) {
                int offset = (y * w_dim + x);
                
                // Classify
                float max_score = -1000.0f;
                int max_id = -1;
                float* cur_cls = cls_data + offset * cls_step;
                for(int c=0; c<config_.class_num; ++c) {
                    if(cur_cls[c] > max_score) { max_score = cur_cls[c]; max_id = c; }
                }
                float score = sigmoid(max_score);
                if (score < config_.conf_thres) continue;

                // Box Decode
                float* cur_box = box_data + offset * box_step;
                float dist[4];
                for(int k=0; k<4; ++k) {
                    float val = 0, sum = 0;
                    for(int r=0; r<config_.reg_max; ++r) {
                        float e = fastExp(cur_box[k*config_.reg_max + r]);
                        val += e * r; sum += e;
                    }
                    dist[k] = val / sum;
                }
                
                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;
                float x1 = pb_cx - dist[0] * stride;
                float y1 = pb_cy - dist[1] * stride;
                float x2 = pb_cx + dist[2] * stride;
                float y2 = pb_cy + dist[3] * stride;

                // Mask Coeffs
                float* cur_msk = msk_data + offset * msk_step;
                std::vector<float> coeffs(cur_msk, cur_msk + config_.num_mask);

                class_ids.push_back(max_id);
                confidences.push_back(score);
                boxes.emplace_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                mask_coeffs.push_back(coeffs);
            }
        }
    }

    // 2. NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config_.conf_thres, config_.nms_thres, indices);
    if (indices.empty()) return;

    // 3. Process Proto Mask
    auto& proto_tensor = tensors.back();
    float* proto_ptr = reinterpret_cast<float*>(proto_tensor->sysMem[0].virAddr);
    int proto_h = 160; 
    int proto_w = 160;
    int proto_c = 32;
    cv::Mat proto_mat(proto_h * proto_w, proto_c, CV_32F, proto_ptr);

    for (int idx : indices) {
        SegResult res;
        res.id = class_ids[idx];
        res.score = confidences[idx];
        res.class_name = (res.id < (int)config_.class_names.size()) ? config_.class_names[res.id] : "unknown";

        // === 坐标还原 (核心逻辑：减 Pad，除 Ratio) ===
        float bx1 = (boxes[idx].x - pad_w_) / ratio_;
        float by1 = (boxes[idx].y - pad_h_) / ratio_;
        float bx2 = (boxes[idx].br().x - pad_w_) / ratio_;
        float by2 = (boxes[idx].br().y - pad_h_) / ratio_;
        res.box = cv::Rect(cv::Point(bx1, by1), cv::Point(bx2, by2));

        // === Mask 计算与还原 ===
        cv::Mat coeff_mat(1, proto_c, CV_32F, mask_coeffs[idx].data());
        cv::Mat mask_raw = proto_mat * coeff_mat.t();
        mask_raw = mask_raw.reshape(1, proto_h);
        cv::exp(-mask_raw, mask_raw);
        mask_raw = 1.0 / (1.0 + mask_raw); // Sigmoid

        // Resize 到模型输入尺寸 (640x640)
        cv::Mat mask_model_size;
        cv::resize(mask_raw, mask_model_size, cv::Size(model_w, model_h));

        // Crop: 去除 Padding 区域
        int valid_w = model_w - 2 * pad_w_;
        int valid_h = model_h - 2 * pad_h_;
        cv::Rect valid_roi(pad_w_, pad_h_, valid_w, valid_h);
        
        // 边界保护
        valid_roi = valid_roi & cv::Rect(0, 0, model_w, model_h);
        if (valid_roi.area() <= 0) continue;

        cv::Mat mask_cropped = mask_model_size(valid_roi).clone();

        res.mask = mask_cropped; 
        
        results.push_back(res);
    }
}

cv::Scalar BPU_Segment::GetColor(int id) {
    int r = (id * 123 + 45) % 255;
    int g = (id * 234 + 90) % 255;
    int b = (id * 345 + 135) % 255;
    return cv::Scalar(b, g, r);
}

void BPU_Segment::draw_detection(cv::Mat& img, const SegResult& res) {
    cv::Scalar color = GetColor(res.id);

    // 1. 绘制 Mask (Resize 到原图大小)
    cv::Mat mask_final;
    if (!res.mask.empty()) {
        cv::resize(res.mask, mask_final, img.size(), 0, 0, cv::INTER_NEAREST);
        
        cv::Mat mask_bin = mask_final > 0.5;
        cv::Mat roi;
        cv::Mat color_mask(img.size(), CV_8UC3, color);
        
        img.copyTo(roi, mask_bin);
        cv::addWeighted(roi, 0.6, color_mask, 0.4, 0.0, roi);
        roi.copyTo(img, mask_bin);
    }

    // 2. 绘制 Box
    cv::rectangle(img, res.box, color, 2);

    // 3. 绘制 Label
    std::string label = res.class_name + " " + std::to_string((int)(res.score * 100)) + "%";
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int label_y = std::max(res.box.y, label_size.height + 10);
    
    cv::rectangle(img, cv::Point(res.box.x, label_y - label_size.height - 5),
                  cv::Point(res.box.x + label_size.width, label_y + 5), color, -1);
    cv::putText(img, label, cv::Point(res.box.x, label_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

void BPU_Segment::detect_result(cv::Mat& img, 
                               const std::vector<SegResult>& results, 
                               double fps,
                               bool show_img) {
    // 1. 逻辑：如果不显示，则尝试关闭窗口，并直接返回（节省画图的 CPU 资源）
    if (!show_img) {
        if (window_created_) {
            // 只有当窗口确实存在时才去销毁
            try {
                cv::destroyWindow("YOLO Seg Visualization");
            } catch (...) {}
            window_created_ = false;
            cv::waitKey(1);
        }
        return; 
    }

    // --- 以下是 show_img = true 时的逻辑 ---

    // 2. 绘制结果 (画框、画 Mask)
    for (const auto& res : results) {
        draw_detection(img, res);
    }

    // 3. 绘制 FPS (只在图像上显示)
    cv::putText(img, "FPS: " + std::to_string((int)fps), cv::Point(20, 50), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    // 4. 显示窗口 (全屏逻辑)
    std::string win_name = "YOLO Seg Visualization";
    try {
        // 检查窗口是否存在 (鲁棒性处理)
        if (!window_created_ || cv::getWindowProperty(win_name, cv::WND_PROP_VISIBLE) < 1.0) {
            cv::namedWindow(win_name, cv::WINDOW_NORMAL);
            cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            window_created_ = true;
        }
    } catch(...) {
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        window_created_ = true;
    }
    
    cv::imshow(win_name, img);
    // 增加一点延时，给 GUI 刷新机会，与 Python 的 waitKey(10) 类似
    cv::waitKey(10); 
}