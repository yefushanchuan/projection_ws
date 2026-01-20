#include "detect_yolov8_11_cpu/cpu_detect.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

// 初始化列表
static const std::vector<std::string> COCO_CLASSES = {
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

CPU_Detect::CPU_Detect(const std::string& model_path, float conf_thres, float iou_thres)
    : env_(ORT_LOGGING_LEVEL_WARNING, "Yolo11"), 
      session_options_(), 
      conf_threshold_(conf_thres), 
      iou_threshold_(iou_thres),
      class_names_(COCO_CLASSES)
{
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        throw;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    
    // Input info
    auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_name_ = std::string(input_name_ptr.get());
    auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_h_ = input_shape[2];
    input_w_ = input_shape[3];

    // Output info
    auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = std::string(output_name_ptr.get());
}

std::vector<yolo_common::UnifiedResult> CPU_Detect::detect(const cv::Mat& img) {
    // 1. 预处理 (Letterbox)
    cv::Mat input_img;
    float ratio = std::min((float)input_w_ / img.cols, (float)input_h_ / img.rows);
    int new_w = std::round(img.cols * ratio);
    int new_h = std::round(img.rows * ratio);
    int dw = (input_w_ - new_w) / 2;
    int dh = (input_h_ - new_h) / 2;

    cv::resize(img, input_img, cv::Size(new_w, new_h));
    cv::copyMakeBorder(input_img, input_img, dh, input_h_ - new_h - dh, 
                       dw, input_w_ - new_w - dw, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat blob;
    cv::dnn::blobFromImage(input_img, blob, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);

    // 2. 推理
    std::vector<int64_t> input_dims = {1, 3, input_h_, input_w_};
    size_t input_tensor_size = 1 * 3 * input_h_ * input_w_;
    
    // 检查连续性
    if (!blob.isContinuous()) {
         return {};
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)blob.datastart, input_tensor_size, input_dims.data(), input_dims.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // 3. 后处理
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // Shape: [Batch, Channels, Anchors] -> [1, 84, 8400]
    int rows = output_shape[2]; // 8400 anchors
    int dimensions = output_shape[1]; // 84 (4 box + 80 cls)

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {            
        float max_score = -1.0f;
        int class_id = -1;
        
        // Find best class score
        // Class scores start at index 4
        for (int c = 0; c < 80; ++c) {
            float score = output_data[(4 + c) * rows + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= conf_threshold_) {
            float cx = output_data[0 * rows + i];
            float cy = output_data[1 * rows + i];
            float w  = output_data[2 * rows + i];
            float h  = output_data[3 * rows + i];

            // Restore coordinates
            float x = (cx - w / 2 - dw) / ratio;
            float y = (cy - h / 2 - dh) / ratio;
            float w_orig = w / ratio;
            float h_orig = h / ratio;

            boxes.push_back(cv::Rect(x, y, w_orig, h_orig));
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, iou_threshold_, indices);

    std::vector<yolo_common::UnifiedResult> results;
    for (int idx : indices) {
        yolo_common::UnifiedResult res;
        res.id = class_ids[idx];
        res.score = confidences[idx];
        res.box = boxes[idx];
        
        // 计算中心点 (UnifiedResult 规范)
        res.center = cv::Point2f(res.box.x + res.box.width / 2.0f, 
                                 res.box.y + res.box.height / 2.0f);
                                 
        if (res.id >= 0 && res.id < (int)class_names_.size()) {
            res.class_name = class_names_[res.id];
        } else {
            res.class_name = "unknown";
        }
        
        // 此包为纯检测，mask 为空，mic_radius 为 0
        results.push_back(res);
    }
    return results;
}