#include "segment/yolo_engine.h"
#include <cmath>
#include <algorithm>

namespace yolo_seg {

inline float fastExp(float x) {
    union { uint32_t i; float f; } v;
    v.i = (12102203.1616540672f * x + 1064807160.56887296f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

YoloEngine::YoloEngine() {
    config_.class_names.resize(80, "object");
    // 这里可以扩展加载 names 文件
    config_.class_names[0] = "person";
}

void YoloEngine::BGRToNV12(const cv::Mat& bgr, cv::Mat& nv12) {
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

void YoloEngine::PreProcess(const cv::Mat& bgr_img, int model_w, int model_h, cv::Mat& nv12_out) {
    // 1. Resize 到模型输入大小
    cv::Mat resized_img;
    cv::resize(bgr_img, resized_img, cv::Size(model_w, model_h));
    
    // 2. 转 NV12
    BGRToNV12(resized_img, nv12_out);
}

void YoloEngine::PostProcess(const std::vector<std::shared_ptr<hobot::dnn_node::DNNTensor>>& tensors,
                             int model_h, int model_w,
                             std::vector<SegResult>& results) {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> mask_coeffs;

    // 解析 Detect Head (Strides 8, 16, 32)
    for (int i = 0; i < 3; ++i) {
        if ((size_t)(i * 3 + 2) >= tensors.size()) break;

        int stride = config_.strides[i];
        auto& cls_tensor = tensors[i * 3];
        auto& box_tensor = tensors[i * 3 + 1];
        auto& msk_tensor = tensors[i * 3 + 2];
        
        // 注意：BPU 输出在这里通常需要根据具体的 tensor layout 处理
        // 此处保留原逻辑的假设
        float* cls_data = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
        float* box_data = reinterpret_cast<float*>(box_tensor->sysMem[0].virAddr);
        float* msk_data = reinterpret_cast<float*>(msk_tensor->sysMem[0].virAddr);

        int h = cls_tensor->properties.validShape.dimensionSize[1]; 
        int w = cls_tensor->properties.validShape.dimensionSize[2]; 
        
        int cls_step = config_.class_num;
        int box_step = 4 * config_.reg_max;
        int msk_step = config_.num_mask;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int offset = (y * w + x);
                
                // 找最大类别分值
                float max_score = -1000.0f;
                int max_id = -1;
                float* cur_cls = cls_data + offset * cls_step;
                
                for(int c=0; c<config_.class_num; ++c) {
                    if(cur_cls[c] > max_score) {
                        max_score = cur_cls[c];
                        max_id = c;
                    }
                }
                
                float score = sigmoid(max_score);
                if (score < config_.score_thres) continue;

                // 解码 Box (DFL)
                float* cur_box = box_data + offset * box_step;
                float dist[4] = {0};
                for(int k=0; k<4; ++k) {
                    float val = 0;
                    float sum = 0;
                    for(int r=0; r<config_.reg_max; ++r) {
                        float e = fastExp(cur_box[k*config_.reg_max + r]);
                        val += e * r;
                        sum += e;
                    }
                    dist[k] = val / sum;
                }
                
                float x1 = (x + 0.5f - dist[0]) * stride;
                float y1 = (y + 0.5f - dist[1]) * stride;
                float x2 = (x + 0.5f + dist[2]) * stride;
                float y2 = (y + 0.5f + dist[3]) * stride;

                // 读取 Mask 系数
                float* cur_msk = msk_data + offset * msk_step;
                std::vector<float> coeffs(cur_msk, cur_msk + config_.num_mask);

                class_ids.push_back(max_id);
                confidences.push_back(score);
                boxes.emplace_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                mask_coeffs.push_back(coeffs);
            }
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config_.score_thres, config_.nms_thres, indices);

    if (indices.empty()) return;

    // 处理 Proto Mask
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
        res.box = boxes[idx];
        res.class_name = config_.class_names[res.id];

        // Mask 矩阵乘法
        cv::Mat coeff_mat(1, proto_c, CV_32F, mask_coeffs[idx].data());
        cv::Mat mask_raw = proto_mat * coeff_mat.t();
        mask_raw = mask_raw.reshape(1, proto_h);
        
        cv::exp(-mask_raw, mask_raw); // Sigmoid part 1
        mask_raw = 1.0 / (1.0 + mask_raw); // Sigmoid part 2
        
        cv::Mat mask_large;
        cv::resize(mask_raw, mask_large, cv::Size(model_w, model_h));
        
        cv::Mat final_mask = cv::Mat::zeros(model_h, model_w, CV_8UC1);
        cv::Mat mask_binary = mask_large > 0.5;
        
        cv::Rect valid_box = res.box & cv::Rect(0, 0, model_w, model_h);
        if (valid_box.area() > 0) {
            mask_binary(valid_box).copyTo(final_mask(valid_box));
        }
        
        res.mask = final_mask;
        results.push_back(res);
    }
}

cv::Scalar YoloEngine::GetColor(int id) {
    int r = (id * 123 + 45) % 255;
    int g = (id * 234 + 90) % 255;
    int b = (id * 345 + 135) % 255;
    return cv::Scalar(b, g, r);
}

void YoloEngine::Visualize(cv::Mat& draw_img, 
                           const std::vector<SegResult>& results, 
                           int model_w, int model_h, 
                           double fps,
                           bool show_window) {
    
    float x_scale = (float)draw_img.cols / model_w;
    float y_scale = (float)draw_img.rows / model_h;

    for (const auto& res : results) {
        cv::Scalar color = GetColor(res.id);

        // Resize Mask to draw_img size
        cv::Mat mask_resized;
        if (res.mask.size() != draw_img.size()) {
            cv::resize(res.mask, mask_resized, draw_img.size(), 0, 0, cv::INTER_NEAREST);
        } else {
            mask_resized = res.mask;
        }

        // Draw Mask
        cv::Mat roi;
        cv::Mat color_mask(draw_img.size(), CV_8UC3, color);
        draw_img.copyTo(roi, mask_resized);
        cv::addWeighted(roi, 0.6, color_mask, 0.4, 0.0, roi);
        roi.copyTo(draw_img, mask_resized);

        // Draw Box
        cv::Rect rect_scaled;
        rect_scaled.x = res.box.x * x_scale;
        rect_scaled.y = res.box.y * y_scale;
        rect_scaled.width = res.box.width * x_scale;
        rect_scaled.height = res.box.height * y_scale;
        
        cv::rectangle(draw_img, rect_scaled, color, 2);

        // Draw Label
        std::string label = res.class_name + " " + std::to_string((int)(res.score * 100)) + "%";
        cv::putText(draw_img, label, cv::Point(rect_scaled.x, rect_scaled.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }

    // FPS
    std::string fps_text = "FPS: " + std::to_string((int)fps);
    cv::putText(draw_img, fps_text, cv::Point(20, 50), 
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);

    // Show Window
    if (show_window) {
        std::string win_name = "YOLOv11 Seg Visualization";
        if (!window_created_) {
            cv::namedWindow(win_name, cv::WINDOW_NORMAL);
            cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            window_created_ = true;
        }
        cv::imshow(win_name, draw_img);
        cv::waitKey(1);
    }
}

} // namespace yolo_seg