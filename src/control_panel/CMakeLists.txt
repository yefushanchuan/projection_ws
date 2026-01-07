cmake_minimum_required(VERSION 3.8)
project(control_panel)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets Core Gui)

# 开启 MOC 处理
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON) # 建议加上这个，如果有资源文件的话

include_directories(include)

# !!! 关键修改 !!!
# 1. 确保所有 .cpp 文件都列在这里
# 2. 最好把带有 Q_OBJECT 宏的 .h 文件也列在这里，方便 AUTOMOC 识别
add_executable(panel_node
  src/main.cpp
  src/mainwindow.cpp
  # 假设你有一个单独的 RosWorker 实现文件，如果是，请取消下面这行的注释：
  # src/rosworker.cpp  
  
  # 显式添加头文件是个好习惯
  include/control_panel/mainwindow.h
  # include/control_panel/rosworker.h
)

target_link_libraries(panel_node
  Qt5::Widgets
  Qt5::Core
  Qt5::Gui
)

ament_target_dependencies(panel_node rclcpp std_msgs)

install(TARGETS panel_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()