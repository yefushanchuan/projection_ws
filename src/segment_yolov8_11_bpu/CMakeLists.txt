cmake_minimum_required(VERSION 3.8)
project(segment_yolov8_11_bpu)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================================================================
# 1. Find Dependencies
# =========================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(dnn_node REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(object3d_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(yolo_common REQUIRED)

# =========================================================================
# 2. TROS/BPU Library Paths
# =========================================================================
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
# 3. Build Executable
# =========================================================================
add_executable(yolo_seg_node 
  src/yolo_seg_node.cpp
  src/bpu_seg_dnn.cpp
)

# =========================================================================
# 4. Include Directories
# =========================================================================
target_include_directories(yolo_seg_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${OpenCV_INCLUDE_DIRS}
  ${RAPIDJSON_INCLUDE_DIRS}
)

# =========================================================================
# 5. Link Dependencies
# =========================================================================
ament_target_dependencies(yolo_seg_node
  "rclcpp"
  "sensor_msgs"
  "cv_bridge"
  "dnn_node"
  "ament_index_cpp"
  "object3d_msgs"
  "message_filters"
  "yolo_common"
)

target_link_libraries(yolo_seg_node
  ${OpenCV_LIBRARIES}
)

# =========================================================================
# 6. Install Rules
# =========================================================================
# 6.1 安装可执行文件
install(TARGETS yolo_seg_node
  DESTINATION lib/${PROJECT_NAME})

# 6.2 安装头文件 (如果有对外暴露的)
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()