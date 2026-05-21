# 01. 사전 요구사항 및 아키텍처 개요

> **NVIDIA Physical AI Stack + TurtleBot3 통합 프로젝트의 사전 준비 사항**

## 1.1 시스템 요구사항

### 1.1.1 개발 호스트 (데스크톱)
Isaac Sim + Isaac Lab + Cosmos 실행을 위한 메인 시스템입니다.

| 구성 요소 | 최소 사양 | 권장 사양 | 비고 |
|-----------|-----------|-----------|------|
| **GPU** | NVIDIA RTX 3060 (12GB VRAM) | **NVIDIA RTX 4090 (24GB)** 이상 | Ampere 이상 아키텍처 필수 |
| **CPU** | Intel i7-12700 / AMD Ryzen 7 5800X | Intel i9-13900K / AMD Ryzen 9 7950X | |
| **RAM** | 32GB | **64GB** | Isaac Lab 대규모 병렬 학습 시 64GB+ |
| **Storage** | 100GB | **500GB NVMe SSD** | 합성 데이터, 체크포인트 저장 |
| **NIC** | 1Gbps | 10Gbps | ROS2 분산 통신 |
| **OS** | **Ubuntu 22.04 LTS** | Ubuntu 22.04 LTS | Windows는 WSL2로 제한적 지원 |

> **⚠️ 중요**: Windows 11에서도 Isaac Sim이 기술적으로 실행 가능하나, CUDA 성능 이슈로 **Ubuntu 22.04 LTS**를 강력 권장합니다.

### 1.1.2 NVIDIA Jetson Orin Nano

| 구성 요소 | 사양 |
|-----------|------|
| **Model** | Jetson Orin Nano Developer Kit (8GB) |
| **JetPack** | JetPack 6.0+ (L4T r36+) |
| **GPU** | 1024-core NVIDIA Ampere |
| **CPU** | 6-core ARM Cortex-A78AE v8.2 |
| **RAM** | 8GB (shared) |
| **Storage** | ≥ 64GB microSD or NVMe SSD |
| **Network** | Wi-Fi 6 + Gigabit Ethernet |
| **Power** | 15W ~ 25W (MAXN 모드) |

### 1.1.3 Raspberry Pi 5 (Dual Robot 용)

| 구성 요소 | 사양 |
|-----------|------|
| **Model** | Raspberry Pi 5 (4GB/8GB) |
| **OS** | Raspberry Pi OS (64-bit) or Ubuntu 24.04 LTS |
| **GPU** | VideoCore VII |
| **CPU** | 4-core ARM Cortex-A76 @ 2.4GHz |
| **ROS2** | ROS2 Jazzy |

### 1.1.4 TurtleBot3 하드웨어

| 구성 요소 | Robot 1 (Jetson) | Robot 2 (RPi) |
|-----------|------------------|---------------|
| **Base** | TurtleBot3 Burger or Waffle Pi | TurtleBot3 Burger |
| **SBC** | **Jetson Orin Nano** | **Raspberry Pi 5** |
| **LiDAR** | 360° LiDAR (LDS-01) | 360° LiDAR (LDS-01) |
| **Camera** | (선택) USB Camera or Intel RealSense | (선택) Raspberry Pi Camera v3 |
| **IMU** | 내장 (Gyro + Accelerometer) | 내장 (Gyro + Accelerometer) |
| **Actuator** | DYNAMIXEL XL430-W250 | DYNAMIXEL XL430-W250 |
| **Power** | OpenCR + 외부 보조 배터리 (5V/3A) | OpenCR + 외부 보조 배터리 |

## 1.2 소프트웨어 의존성

### 1.2.1 메인 시스템 (Ubuntu 22.04)

```bash
# 필수 시스템 패키지
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-venv \
    wget \
    curl \
    vim \
    net-tools \
    can-utils \
    libssl-dev \
    libeigen3-dev \
    libboost-all-dev \
    libzmq3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libatlas-base-dev

# NVIDIA 관련
sudo apt install -y \
    nvidia-driver-545 \
    nvidia-utils-545 \
    libcudnn8 \
    libcudnn8-dev

# CUDA (12.4+)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

### 1.2.2 ROS2 Humble 설치 (메인 시스템)

```bash
# ROS2 Humble 설치
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions

# ROS2 빌드 도구
sudo apt install -y \
    python3-rosdep \
    python3-argcomplete \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-turtlebot3* \
    ros-humble-robot-localization \
    ros-humble-teleop-twist-keyboard \
    ros-humble-xacro
```

### 1.2.3 NVIDIA Isaac Sim 요구사항

```bash
# NVIDIA Container Toolkit 설치 (Docker 기반 권장)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Docker Compose
sudo apt install -y docker-compose-plugin
```

## 1.3 시스템 아키텍처

### 1.3.1 Phase 1: 단일 로봇 (Jetson Orin Nano)

```
┌──────────────────────────┐     ROS2 DDS (Ethernet/WiFi)     ┌──────────────────────┐
│   Desktop (Ubuntu 22.04) │◄──────────────────────────────►  │  TurtleBot3 (Jetson) │
│                          │    /cmd_vel, /odom, /scan        │                      │
│  ┌───────────────────┐   │                                   │  ┌────────────────┐  │
│  │   Isaac Sim       │   │                                   │  │ Nav2 (C++)     │  │
│  │   Isaac Lab       │   │                                   │  │ Policy Node    │  │
│  │   Cosmos          │   │                                   │  │ LiDAR Driver   │  │
│  └───────────────────┘   │                                   │  │ Motor Driver   │  │
│          │               │                                   │  └────────────────┘  │
│          ▼               │                                   │                      │
│  ┌───────────────────┐   │                                   └──────────────────────┘
│  │ Policy (.onnx)    │   │
│  └───────────────────┘   │
└──────────────────────────┘
```

### 1.3.2 Phase 4: 듀얼 로봇 (확장)

```
┌────────────────────────────────────────────────────────────────────┐
│                      Desktop (Ubuntu 22.04)                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │ Isaac Sim  │  │ Isaac Lab  │  │  Cosmos    │  │  Central   │   │
│  │ (Multi-Rbt)│  │(Multi-Agt) │  │  Pipeline  │  │  Nav Server│   │
│  └─────┬──────┘  └────────────┘  └────────────┘  └──────┬─────┘   │
│        │                                                  │        │
└────────┼──────────────────────────────────────────────────┼────────┘
         │ ROS2 DDS (namespace: /tb1)                       │ ROS2 DDS (/tb2)
         ▼                                                  ▼
┌──────────────────────┐                     ┌──────────────────────┐
│  TurtleBot3 #1       │                     │  TurtleBot3 #2       │
│  (Jetson Orin Nano)  │                     │  (Raspberry Pi 5)    │
│──────────────────────│                     │──────────────────────│
│ /tb1/scan            │                     │ /tb2/scan            │
│ /tb1/cmd_vel         │                     │ /tb2/cmd_vel         │
│ /tb1/odom            │                     │ /tb2/odom            │
│ /tb1/policy/action   │                     │ /tb2/policy/action   │
│                      │                     │                      │
│ • 고성능 정책 추론    │                     │ • 경량 Nav2 실행     │
│ • TensorRT 최적화     │                     │ • LiDAR SLAM         │
│ • 카메라 비전 처리    │                     │ • 원격 명령 기반     │
└──────────────────────┘                     └──────────────────────┘
```

## 1.4 ROS2 네임스페이스 설계

### 단일 로봇 (Phase 1-3)
기본 토픽 사용 (네임스페이스 없음):
- `/cmd_vel` - 속도 명령
- `/odom` - 오도메트리
- `/scan` - LiDAR 스캔
- `/map` - 점유 그리드 맵
- `/tf` / `/tf_static` - 좌표 변환

### 듀얼 로봇 (Phase 4)
네임스페이스 기반 격리:
```
/tb1/cmd_vel, /tb1/odom, /tb1/scan, /tb1/map
/tb2/cmd_vel, /tb2/odom, /tb2/scan, /tb2/map
```

## 1.5 하드웨어 연결 구성

### Jetson Orin Nano ↔ TurtleBot3 Wiring

```
┌──────────────────────────┐
│    Jetson Orin Nano      │
│                          │
│  GPIO Header (40-pin)    │
│  ┌────────────────────┐  │
│  │ Pin 8  (UART1 TX)  ├──┼──► OpenCR Board (UART RX)
│  │ Pin 10 (UART1 RX)  ├──┼──► OpenCR Board (UART TX)
│  │ Pin 6  (GND)       ├──┼──► OpenCR GND
│  │ Pin 12 (I2S_CLK)   ├──┼──► (Reserved)
│  └────────────────────┘  │
│                          │
│  USB 3.0 Port           │
│  ┌────────────────────┐  │
│  │ USB-A              ├──┼──► LiDAR LDS-01 (USB)
│  └────────────────────┘  │
│                          │
│  CSI Camera Port        │
│  ┌────────────────────┐  │
│  │ CSI-0             ├──┼──► (선택) Camera Module
│  └────────────────────┘  │
└──────────────────────────┘
```

> **참고**: TurtleBot3 Burger의 표준 설정은 OpenCR 보드가 모터 제어를 담당합니다. Jetson Orin Nano는 USB로 LiDAR를 연결하고, UART로 OpenCR과 통신합니다.

### Raspberry Pi 5 ↔ TurtleBot3 Wiring
표준 TurtleBot3 Burger 설정과 동일:
```
┌──────────────────────┐
│   Raspberry Pi 5      │
│                       │
│  GPIO 14 (UART0 TX) ├──► OpenCR Board (UART RX)
│  GPIO 15 (UART0 RX) ├──► OpenCR Board (UART TX)
│  GND                ├──► OpenCR GND
│                       │
│  USB 3.0             │
│  USB-A              ├──► LiDAR LDS-01 (USB)
└──────────────────────┘
```

## 1.6 네트워크 구성

```
┌────────────────────────────────────────────────────┐
│                   ROS2 Domain                       │
│              (DDS: FastDDS / CycloneDDS)            │
│                                                      │
│  Desktop (10.0.0.10)                                │
│    ├── ROS_DOMAIN_ID=42                              │
│    │                                                 │
│  Jetson Orin Nano (10.0.0.20)                       │
│    ├── ROS_DOMAIN_ID=42                              │
│    │  ROS_LOCALHOST_ONLY=0                           │
│    │                                                 │
│  Raspberry Pi 5 (10.0.0.30)                         │
│    ├── ROS_DOMAIN_ID=42                              │
│       ROS_LOCALHOST_ONLY=0                           │
└────────────────────────────────────────────────────┘
```

Wi-Fi 네트워크 환경에서는 CycloneDDS 권장 (FastDDS는 WiFi 환경에서 패킷 손실에 취약):

```bash
# CycloneDDS 설정
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///path/to/cyclonedds.xml
```

## 1.7 개발 워크플로우 개요

```
Step 1: URDF 준비
   ↓
Step 2: Isaac Sim → USD 변환 + 시뮬레이션 환경 구성
   ↓
Step 3: Cosmos 합성 데이터 생성 (선택사항)
   ↓
Step 4: Isaac Lab RL 학습 환경 정의
   ↓
Step 5: PPO/SAC 학습 실행
   ↓
Step 6: Policy 검증 (Isaac Sim 내 평가)
   ↓
Step 7: Policy Export → ONNX → TensorRT
   ↓
Step 8: Jetson Orin Nano 배포
   ↓
Step 9: 실제 주행 테스트
   ↓
Step 10: Dual Robot 확장 (선택사항)
```

## 1.8 참고 자료

- [NVIDIA Isaac Lab Quickstart](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/quickstart.html)
- [NVIDIA Isaac Sim ROS2 Tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/index.html)
- [NVIDIA Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/)
- [TurtleBot3 ROS2 eManual](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)
- [ROS2 Humble Installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [Jetson Orin Nano Developer Kit Guide](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide)
