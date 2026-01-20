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

# 【修改点 1】必须查找 yolo_common
find_package(yolo_common REQUIRED)

# =========================================================
# ARM 架构下手动指定 ONNX Runtime
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
# 2. 编译推理引擎库 (Lib)
# =========================================================
add_library(yolo_detector_lib src/cpu_detect.cpp)

target_include_directories(yolo_detector_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

# 链接原生库 (OpenCV, ONNX)
target_link_libraries(yolo_detector_lib
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIBRARIES}
)

# 【修改点 2】使用 ament 添加 ROS 包依赖 (包含 yolo_common)
# 这会自动处理 yolo_common 的头文件路径，解决找不到头文件的问题
ament_target_dependencies(yolo_detector_lib
  "rclcpp" 
  "yolo_common" 
)

# =========================================================
# 3. 编译 ROS 节点 (Node)
# =========================================================
add_executable(yolo_detect_node src/yolo_detect_node.cpp)

set_target_properties(yolo_detect_node PROPERTIES
  INSTALL_RPATH "${ONNX_INSTALL_PATH}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

target_include_directories(yolo_detect_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

# 链接我们自己的库 和 OpenCV
target_link_libraries(yolo_detect_node
  yolo_detector_lib
  ${OpenCV_LIBRARIES}
)

# 【修改点 3】在依赖列表中添加 yolo_common
ament_target_dependencies(yolo_detect_node
  "rclcpp"
  "sensor_msgs"
  "cv_bridge"
  "message_filters"
  "image_transport"
  "object3d_msgs"
  "ament_index_cpp"
  "yolo_common"   # <--- 加上这个，节点才能使用 visualization.hpp
)

# =========================================================
# 4. 安装规则
# =========================================================
install(TARGETS yolo_detect_node
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS yolo_detector_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()