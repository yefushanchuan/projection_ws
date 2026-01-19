cmake_minimum_required(VERSION 3.8)
project(detect11)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 寻找 ROS 依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(object3d_msgs REQUIRED) # 你的自定义消息包
find_package(ament_index_cpp REQUIRED)

# 2. 寻找外部依赖 (OpenCV 和 ONNX Runtime)
find_package(OpenCV REQUIRED)

set(ONNX_INSTALL_PATH "/opt/onnxruntime")

# 2. 指定头文件路径
set(ONNXRUNTIME_INCLUDE_DIRS 
    "${ONNX_INSTALL_PATH}/include"
)

# 3. 指定库文件路径 (查找 libonnxruntime.so)
find_library(ONNXRUNTIME_LIBRARIES onnxruntime 
    PATHS "${ONNX_INSTALL_PATH}/lib"
    NO_DEFAULT_PATH
)

# 4. 检查是否找到
if (NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNX_INSTALL_PATH}. Please run the download commands.")
else()
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")
endif()

# 3. 定义节点可执行文件
add_executable(detect11_node src/yolo_detect_node.cpp)

# 4. 包含头文件路径
target_include_directories(detect11_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

# 5. 链接库
target_link_libraries(detect11_node
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIBRARIES}
)

# 6. 设置 ROS 依赖
ament_target_dependencies(detect11_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
  image_transport
  ament_index_cpp
  object3d_msgs
)

# 7. 安装规则
install(TARGETS detect11_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()