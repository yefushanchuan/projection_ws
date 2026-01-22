cmake_minimum_required(VERSION 3.8)
project(detect_yolov8_11_cpu)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# =========================================================
# 1. 依赖查找
# =========================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(object3d_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yolo_common REQUIRED)

# =========================================================
# 2. ONNX Runtime 设置
# =========================================================
set(ONNX_INSTALL_PATH "/opt/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNX_INSTALL_PATH}/include")
find_library(ONNXRUNTIME_LIBRARIES onnxruntime PATHS "${ONNX_INSTALL_PATH}/lib" NO_DEFAULT_PATH)

if (NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR "ONNX Runtime not found!")
else()
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")
endif()

# =========================================================
# 3. 编译可执行文件
# =========================================================
add_executable(yolo_detect_node 
  src/yolo_detect_node.cpp
  src/cpu_detect.cpp
)

# =========================================================
# 4. 包含目录
# =========================================================
target_include_directories(yolo_detect_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

# =========================================================
# 5. 链接依赖
# =========================================================
ament_target_dependencies(yolo_detect_node
  "rclcpp"
  "sensor_msgs"
  "cv_bridge"
  "message_filters"
  "image_transport"
  "object3d_msgs"
  "ament_index_cpp"
  "yolo_common"
)

target_link_libraries(yolo_detect_node
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIBRARIES}
)

# =========================================================
# 6. 安装规则
# =========================================================
# 6.1 安装可执行文件
install(TARGETS yolo_detect_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()