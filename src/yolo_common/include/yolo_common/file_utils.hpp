#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace yolo_common {
namespace utils {

// 从文本文件加载类别名，每行一个
inline std::vector<std::string> LoadClassesFromFile(const std::string& file_path) {
    std::vector<std::string> classes;
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        std::cerr << "[YoloCommon] Error: Cannot open file " << file_path << std::endl;
        return {};
    }
    std::string line;
    while (std::getline(ifs, line)) {
        // 去除可能的 Windows 换行符 \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) {
            classes.push_back(line);
        }
    }
    return classes;
}

} // namespace utils
} // namespace yolo_common