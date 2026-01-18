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
    config_.class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    // 自动更新类别数量，防止 config_ 里写的 80 和这里不一致
    config_.class_num = config_.class_names.size();
}

BPU_Segment::~BPU_Segment() {
    if (window_created_) {
        cv::destroyAllWindows(); // 退出时强制关闭所有 OpenCV 窗口
    }
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

        // === 1. 获取模型尺度下的 Box (含 Padding) ===
        // 这里的 boxes[idx] 是基于 640x640 的坐标
        cv::Rect model_box = boxes[idx];
        
        // 边界保护：防止 box 超出模型输入范围
        model_box = model_box & cv::Rect(0, 0, model_w, model_h);
        if (model_box.area() <= 0) continue;

        // === 2. 还原最终坐标 (用于显示) ===
        // 公式：(x - pad) / ratio
        float bx1 = (model_box.x - pad_w_) / ratio_;
        float by1 = (model_box.y - pad_h_) / ratio_;
        float bx2 = (model_box.br().x - pad_w_) / ratio_;
        float by2 = (model_box.br().y - pad_h_) / ratio_;
        res.box = cv::Rect(cv::Point(bx1, by1), cv::Point(bx2, by2));

        // === 3. Mask 计算与裁切 (核心优化) ===
        cv::Mat coeff_mat(1, proto_c, CV_32F, mask_coeffs[idx].data());
        cv::Mat mask_raw = proto_mat * coeff_mat.t(); // (160x160) x 1
        mask_raw = mask_raw.reshape(1, proto_h);      // 160x160
        cv::exp(-mask_raw, mask_raw);
        mask_raw = 1.0 / (1.0 + mask_raw); // Sigmoid

        // Resize 到模型输入尺寸 (640x640)
        // 优化：其实可以不resize全图，但为了对齐精度，通常先resize再crop
        cv::Mat mask_model_full;
        cv::resize(mask_raw, mask_model_full, cv::Size(model_w, model_h)); // 这里可以用 INTER_LINEAR

        // **核心：直接裁切出 ROI Mask**
        // res.mask 现在只包含该物体的小方块区域
        res.mask = mask_model_full(model_box).clone();

        // === 4. 计算最大内切圆 (MIC) ===
        // 因为 res.mask 已经是 ROI 了，直接算即可，速度很快
        if (!res.mask.empty()) {
            // 二值化处理，确保计算距离变换准确
            cv::Mat mask_bin;
            // 注意：mask_model_full 是 float 0~1，这里阈值设 0.5
            cv::threshold(res.mask, mask_bin, 0.5, 1.0, cv::THRESH_BINARY);
            mask_bin.convertTo(mask_bin, CV_8UC1);

            cv::Mat dist_map;
            cv::distanceTransform(mask_bin, dist_map, cv::DIST_L2, cv::DIST_MASK_PRECISE);
            
            double max_val;
            cv::Point max_loc; // 这是相对于 ROI (model_box) 左上角的坐标
            cv::minMaxLoc(dist_map, nullptr, &max_val, nullptr, &max_loc);

            // 还原 Radius 到原图尺度
            res.mic_radius = (float)max_val / ratio_;

            // 还原 Center 到原图尺度
            // 逻辑：(ROI左上角 + ROI内部偏移 - Padding) / ratio
            float center_x = (model_box.x + max_loc.x - pad_w_) / ratio_;
            float center_y = (model_box.y + max_loc.y - pad_h_) / ratio_;
            res.mic_center = cv::Point2f(center_x, center_y);
        } else {
            res.mic_radius = 0;
            res.mic_center = cv::Point2f(0, 0);
        }

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

    // === 1. 绘制 Mask (优化版) ===
    if (!res.mask.empty()) {
        // 安全检查：防止 res.box 超出图像边界
        cv::Rect valid_box = res.box & cv::Rect(0, 0, img.cols, img.rows);
        
        if (valid_box.area() > 0) {
            // 此时 res.mask 是 float 类型的 ROI 小图
            cv::Mat mask_roi_resized;
            
            // 将 ROI Mask 缩放到最终检测框的大小
            // 注意：如果 valid_box 比 res.box 小（即框在图像边缘），需要相应裁剪 mask
            // 为了简单，这里假设绝大多数情况 box 都在图内，或者简单的缩放:
            cv::resize(res.mask, mask_roi_resized, res.box.size());

            // 处理边缘情况：如果 box 被图像边缘截断，我们需要截断 mask
            int x_offset = valid_box.x - res.box.x;
            int y_offset = valid_box.y - res.box.y;
            cv::Rect mask_crop_rect(x_offset, y_offset, valid_box.width, valid_box.height);
            mask_roi_resized = mask_roi_resized(mask_crop_rect);

            // 二值化
            cv::Mat mask_bin = mask_roi_resized > 0.5;

            // 混合颜色 (只在 valid_box 区域操作，速度极快)
            cv::Mat img_roi = img(valid_box);
            cv::Mat color_layer(img_roi.size(), CV_8UC3, color);
            
            // 使用 mask_bin 作为掩码进行混合
            // img_roi = img_roi * 0.6 + color * 0.4
            cv::Mat temp;
            cv::addWeighted(img_roi, 0.6, color_layer, 0.4, 0.0, temp);
            temp.copyTo(img_roi, mask_bin);
        }
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
    std::string win_name = "Segment Result";

    // 1. 如果不显示：关闭窗口并退出
    if (!show_img) {
        if (window_created_) {
            try {
                cv::destroyWindow(win_name);
            } catch (...) {}
            window_created_ = false;
            // 必须处理一次事件循环，否则窗口可能关不掉
            cv::waitKey(1); 
        }
        return; 
    }

    // --- 以下是 show_img = true 时的逻辑 ---

    // 2. 绘制结果 (因为 show_img 为 true，所以必须画)
    if (img.empty()) return;
    
    for (const auto& res : results) {
        draw_detection(img, res);
    }

    // 绘制 FPS
    cv::putText(img, "FPS: " + std::to_string((int)fps), cv::Point(20, 50), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    // 3. 窗口状态检查 (核心修改点：自动复活逻辑)
    // 检查窗口是否被用户手动关闭 (visible < 1.0)
    try {
        if (window_created_ && cv::getWindowProperty(win_name, cv::WND_PROP_VISIBLE) < 1.0) {
            // 关键点：如果检测到窗口被关闭，不是 return，而是重置标志位
            window_created_ = false; 
        }
    } catch (...) {
        // 如果获取属性报错，也认为窗口没了，需要重建
        window_created_ = false;
    }

    // 4. 创建窗口 (如果未创建，或者上面被重置为 false)
    if (!window_created_) {
        try {
            cv::namedWindow(win_name, cv::WINDOW_NORMAL);
            cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            window_created_ = true;
        } catch(...) {
            // 创建失败暂时忽略，下一帧再试
        }
    }
        
    // 5. 显示图像
    if (window_created_) {
        cv::imshow(win_name, img);
        cv::waitKey(1); // 1ms 延时，保持窗口响应
    }
}