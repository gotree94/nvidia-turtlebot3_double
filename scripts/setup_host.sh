#!/bin/bash
# ============================================
# NVIDIA TurtleBot3 Project - Host Setup Script
# Ubuntu 22.04 LTS
# ============================================
# Usage: bash scripts/setup_host.sh

set -e

echo "=============================================="
echo "NVIDIA TurtleBot3 Project - Host Setup"
echo "=============================================="

# ========== System Packages ==========
echo "[1/8] Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-venv \
    wget curl \
    vim net-tools \
    libssl-dev \
    libeigen3-dev \
    libboost-all-dev \
    libzmq3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# ========== NVIDIA Driver ==========
echo "[2/8] Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA driver..."
    sudo apt install -y nvidia-driver-545
    echo "⚠️  REBOOT REQUIRED after driver installation"
else
    echo "NVIDIA driver found:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
fi

# ========== CUDA ==========
echo "[3/8] Installing CUDA 12.4..."
if ! command -v nvcc &> /dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-4
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
else
    echo "CUDA found: $(nvcc --version | grep release)"
fi

# ========== Docker ==========
echo "[4/8] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out/in for group changes."
fi

# ========== NVIDIA Container Toolkit ==========
echo "[5/8] Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Test GPU in Docker
echo "Testing GPU access in Docker..."
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# ========== ROS2 Humble ==========
echo "[6/8] Installing ROS2 Humble..."
if [ ! -d "/opt/ros/humble" ]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
        sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
    sudo apt install -y ros-humble-desktop python3-colcon-common-extensions
    sudo apt install -y \
        ros-humble-gazebo-ros-pkgs \
        ros-humble-navigation2 ros-humble-nav2-bringup \
        ros-humble-turtlebot3* \
        ros-humble-robot-localization \
        ros-humble-teleop-twist-keyboard \
        ros-humble-slam-toolbox
fi

echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc

# ========== Miniconda ==========
echo "[7/8] Installing Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    $HOME/miniconda3/bin/conda init bash
fi

source ~/.bashrc

# ========== Isaac Lab Environment ==========
echo "[8/8] Creating Isaac Lab conda environment..."
conda create -n isaaclab python=3.10 -y
conda activate isaaclab
pip install --upgrade pip
pip install \
    torch>=2.1.0 \
    tensorboard \
    wandb \
    gymnasium \
    hydra-core \
    pyyaml \
    numpy \
    matplotlib \
    opencv-python \
    pillow

# Cosmos environment
conda create -n cosmos python=3.10 -y
conda activate cosmos
pip install --upgrade pip

echo ""
echo "=============================================="
echo "✅ Host setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. docker pull nvcr.io/nvidia/isaac-sim:2025.2.0"
echo "  2. conda activate isaaclab && cd ~/IsaacLab && ./isaaclab.sh --install"
echo "  3. mkdir -p ~/tb3_ws/src"
echo "  4. cd ~/tb3_ws/src && git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git"
echo "  5. cd ~/tb3_ws && colcon build --symlink-install"
echo "  6. Reboot for NVIDIA driver"
echo "=============================================="
