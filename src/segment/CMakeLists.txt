cmake_minimum_required(VERSION 3.8)
project(segment)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(dnn_node REQUIRED)
find_package(OpenCV REQUIRED)

if(DEFINED ENV{TROS_DISTRO})
    set(TROS_PATH /opt/tros/humble)
else()
    set(TROS_PATH /opt/tros/humble) # 默认路径
endif()

link_directories(
  ${TROS_PATH}/lib/
  /usr/hobot/lib/
  /usr/lib/hbbpu/                 
  /usr/lib/hbmedia/               
  /usr/lib/aarch64-linux-gnu/     
)

add_executable(yolo_seg_node src/yolo_seg_node.cpp)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${RAPIDJSON_INCLUDE_DIRS}
)

ament_target_dependencies(yolo_seg_node
  rclcpp
  sensor_msgs
  cv_bridge
  dnn_node
)

target_link_libraries(yolo_seg_node
  ${OpenCV_LIBRARIES}
)

# 安装规则
install(TARGETS yolo_seg_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
