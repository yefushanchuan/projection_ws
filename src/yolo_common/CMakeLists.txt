cmake_minimum_required(VERSION 3.8)
project(yolo_common)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)

# 导出头文件路径
ament_export_include_directories(include)
ament_export_dependencies(OpenCV)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_package()