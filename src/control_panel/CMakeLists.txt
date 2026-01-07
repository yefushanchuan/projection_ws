cmake_minimum_required(VERSION 3.8)
project(control_panel)

# --- 1. 设置 C++ 标准 ---
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- 2. 查找依赖 ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets Core Gui)

# --- 3. 开启 Qt MOC 自动化 ---
# 这会自动处理 Q_OBJECT 宏
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

include_directories(include)

# --- 4. 添加可执行文件 ---
# 必须把 .h 和 .cpp 都列出来，尤其是带有 Q_OBJECT 的 .h
add_executable(panel_node
  src/main.cpp
  src/mainwindow.cpp
  src/rosworker.cpp
  
  include/control_panel/mainwindow.h
  include/control_panel/rosworker.h
)

# --- 5. 链接库 ---
target_link_libraries(panel_node
  Qt5::Widgets
  Qt5::Core
  Qt5::Gui
)

ament_target_dependencies(panel_node rclcpp std_msgs)

# --- 6. 安装 ---
install(TARGETS panel_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()