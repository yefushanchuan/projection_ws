cmake_minimum_required(VERSION 3.8)
project(image_viewer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(object3d_msgs REQUIRED)

add_executable(image_viewer_talker src/image_viewer_talker.cpp)
add_executable(image_viewer_listener src/image_viewer_listener.cpp)

target_include_directories(image_viewer_talker PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(image_viewer_talker PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  image_viewer_talker
  "rclcpp"
  "cv_bridge"
  "sensor_msgs"
  "object3d_msgs"
)
ament_target_dependencies(
  image_viewer_listener
  "rclcpp"
  "cv_bridge"
  "sensor_msgs"
  "object3d_msgs"
)
target_link_libraries(image_viewer_talker ${OpenCV_LIBS})
target_link_libraries(image_viewer_listener ${OpenCV_LIBS})

install(TARGETS image_viewer_talker
image_viewer_listener
  DESTINATION lib/${PROJECT_NAME})

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
