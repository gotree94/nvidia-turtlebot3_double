#!/bin/bash
# ============================================
# Jetson Orin Nano Setup Script
# JetPack 6.0 / Ubuntu 22.04
# ============================================
# Usage: bash scripts/setup_jetson.sh
# Run on Jetson Orin Nano

set -e

echo "=============================================="
echo "Jetson Orin Nano - TurtleBot3 Setup"
echo "=============================================="

# ========== System Update ==========
echo "[1/7] System update..."
sudo apt update && sudo apt upgrade -y

# ========== Performance Mode ==========
echo "[2/7] Setting MAXN performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# SWAP
if [ ! -f /swapfile ]; then
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# ========== Development Tools ==========
echo "[3/7] Installing development tools..."
sudo apt install -y \
    python3-pip python3-venv \
    git cmake vim \
    can-utils \
    net-tools

# CUDA env
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# ========== ROS2 Humble ==========
echo "[4/7] Installing ROS2 Humble..."
if [ ! -d "/opt/ros/humble" ]; then
    sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
        sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
    sudo apt install -y ros-humble-desktop python3-colcon-common-extensions
fi

echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
source ~/.bashrc

# ========== TurtleBot3 Packages ==========
echo "[5/7] Building TurtleBot3 packages..."
mkdir -p ~/tb3_ws/src
cd ~/tb3_ws/src

if [ ! -d "turtlebot3" ]; then
    git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git
    git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
    git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
    git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
fi

cd ~/tb3_ws
colcon build --symlink-install
echo 'source ~/tb3_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc

# ========== Python Packages ==========
echo "[6/7] Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    torch torchvision \
    numpy opencv-python \
    matplotlib scipy \
    onnxruntime-gpu \
    rclpy \
    sensor_msgs geometry_msgs nav_msgs \
    tf_transformations \
    transforms3d \
    pyyaml

# ========== UDEV Rules ==========
echo "[7/7] Setting up udev rules..."
# OpenCR
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11", SYMLINK+="opencr", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/99-opencr.rules
# LiDAR LDS-01
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="lidar", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/99-lidar.rules

sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "=============================================="
echo "✅ Jetson Orin Nano setup complete!"
echo "=============================================="
echo ""
echo "Quick test:"
echo "  ros2 topic list"
echo "  export TURTLEBOT3_MODEL=burger"
echo "  ros2 launch turtlebot3_bringup turtlebot3_robot.launch.py"
echo "=============================================="
