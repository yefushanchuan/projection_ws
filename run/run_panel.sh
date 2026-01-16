#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}          正在启动 Control Panel             ${NC}"
echo -e "${GREEN}=============================================${NC}"

echo "[1/5] 正在加载 ROS Humble 环境..."
source /opt/ros/humble/setup.bash

echo "[2/5] 正在加载 TROS 环境..."
source /opt/tros/humble/setup.bash

echo "[3/5] 正在加载 Realsense 工作空间..."
source /home/sunrise/realsense_ws/install/setup.bash

echo "[4/5] 正在加载 Projection 工作空间..."
source /home/sunrise/projection_ws/install/setup.bash

echo "[5/5] 正在启动 GUI 界面..."
# 启动你的节点
ros2 run control_panel control_panel_node