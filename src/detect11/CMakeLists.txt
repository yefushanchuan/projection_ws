cmake_minimum_required(VERSION 3.8)
project(detect11)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 1. 依赖查找
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(object3d_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)

# ARM 架构下手动指定 ONNX Runtime (根据你之前的成功经验)
set(ONNX_INSTALL_PATH "/opt/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNX_INSTALL_PATH}/include")
find_library(ONNXRUNTIME_LIBRARIES onnxruntime PATHS "${ONNX_INSTALL_PATH}/lib" NO_DEFAULT_PATH)

if (NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR "ONNX Runtime not found!")
else()
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")
endif()

# 2. 【核心修改】编译推理引擎库 (Lib)
# 将 src/cpu_detect.cpp 编译为一个名为 yolo_detector_lib 的库
add_library(yolo_detector_lib src/cpu_detect.cpp)

# 设置库的头文件路径
target_include_directories(yolo_detector_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

# 库需要链接 OpenCV 和 ONNXRuntime
target_link_libraries(yolo_detector_lib
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIBRARIES}
)

# 3. 【核心修改】编译 ROS 节点 (Node)
# 将 src/yolo_detect_node.cpp 编译为可执行文件
add_executable(detect11_node src/yolo_detect_node.cpp)

# 设置节点的头文件路径
target_include_directories(detect11_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

# 节点需要链接：刚才编译的库(yolo_detector_lib) + ROS 组件
target_link_libraries(detect11_node
  yolo_detector_lib   # <--- 链接我们自己写的库
  ${OpenCV_LIBRARIES}
  ament_index_cpp
)

# 添加 ROS 依赖
ament_target_dependencies(detect11_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
  image_transport
  object3d_msgs
)

# 4. 安装规则
# 安装可执行文件
install(TARGETS detect11_node
  DESTINATION lib/${PROJECT_NAME})

# 安装库 (为了规范，虽然这个包不一定给别人用)
install(TARGETS yolo_detector_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include)

# 安装模型文件
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()