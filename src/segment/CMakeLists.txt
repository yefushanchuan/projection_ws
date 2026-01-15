cmake_minimum_required(VERSION 3.8)
project(segment)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(dnn_node REQUIRED)
find_package(OpenCV REQUIRED)

# 2. TROS/BPU Library Paths
if(DEFINED ENV{TROS_DISTRO})
    set(TROS_PATH /opt/tros/humble)
else()
    set(TROS_PATH /opt/tros/humble)
endif()

link_directories(
  ${TROS_PATH}/lib/
  /usr/hobot/lib/
  /usr/lib/hbbpu/                 
  /usr/lib/hbmedia/               
  /usr/lib/aarch64-linux-gnu/     
)

# =========================================================================
# 3. Build Executable (核心修改)
# 将 yolo_seg_node.cpp 和 bpu_seg_hobot_dnn.cpp 一起编译
# =========================================================================
add_executable(yolo_seg_node 
  src/yolo_seg_node.cpp
  src/bpu_seg_hobot_dnn.cpp
)

# 4. Include Directories
# "include" 目录包含了 include/segment/bpu_seg_hobot_dnn.h
# 代码中可以使用 #include "segment/bpu_seg_hobot_dnn.h"
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${RAPIDJSON_INCLUDE_DIRS}
)

# 5. Link Dependencies
ament_target_dependencies(yolo_seg_node
  rclcpp
  sensor_msgs
  cv_bridge
  dnn_node
)

target_link_libraries(yolo_seg_node
  ${OpenCV_LIBRARIES}
)

# =========================================================================
# 6. Install Rules
# =========================================================================
install(TARGETS yolo_seg_node
  DESTINATION lib/${PROJECT_NAME})

# 安装头文件和模型文件夹
install(DIRECTORY include models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()