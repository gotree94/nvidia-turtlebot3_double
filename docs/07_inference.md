# 07. 추론 및 Jetson Orin Nano 배포

> **학습된 Policy를 Jetson Orin Nano에 배포하여 실시간 자율주행**

## 7.1 배포 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                     Desktop (Train)                               │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │  Isaac Lab 학습 → Policy (.pt) → ONNX → TensorRT (.plan) │    │
│  └───────────────────────────────────────────────────────────┘    │
│                              │ SCP / USB                          │
└──────────────────────────────┼────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Jetson Orin Nano (Run)                         │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │  ROS2 Node: policy_node.py                                │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │    │
│  │  │  TensorRT     │  │  LiDAR       │  │  Motor         │  │    │
│  │  │  Inference    │◄─┤  Preprocess  ├──►  Control       │  │    │
│  │  │  (C++/Python) │  │              │  │  (/cmd_vel)    │  │    │
│  │  └──────────────┘  └──────────────┘  └────────────────┘  │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │  Navigation2 Stack (선택적 Fallback)                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │    │
│  │  │ Global   │ │ Local    │ │ Behavior │ │ Lifecycle  │  │    │
│  │  │ Planner  │ │ Planner  │ │ Tree     │ │ Manager    │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## 7.2 Jetson Orin Nano 설정

### 7.2.1 시스템 설정

```bash
# ============================================
# Jetson Orin Nano 초기 설정
# ============================================

# 1. 성능 모드
sudo nvpmodel -m 0          # MAXN 모드 (15W ~ 25W)
sudo jetson_clocks           # 모든 클럭 최대

# 2. SWAP 파일 (메모리 부족 방지)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 3. ROS2 Humble 설치
sudo apt update && sudo apt install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-teleop-twist-keyboard

# 4. 환경 변수 설정
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
source ~/.bashrc
```

### 7.2.2 TurtleBot3 패키지 설치

```bash
# TurtleBot3 ROS2 패키지
mkdir -p ~/tb3_ws/src
cd ~/tb3_ws/src

git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git

cd ~/tb3_ws
colcon build --symlink-install
echo 'source ~/tb3_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

### 7.2.3 Python 추론 환경

```bash
# Python 패키지
pip3 install --upgrade pip
pip3 install \
    torch \
    torchvision \
    numpy \
    opencv-python \
    matplotlib \
    scipy \
    onnxruntime-gpu \
    tensorrt \
    rclpy \
    sensor_msgs \
    geometry_msgs \
    nav_msgs \
    tf_transformations
```

## 7.3 TensorRT 추론 노드

**`src/deployment/jetson_inference_node.py`**:

```python
#!/usr/bin/env python3
"""
TurtleBot3 RL Policy Inference Node for Jetson Orin Nano

이 노드는 TensorRT 최적화된 정책을 사용하여 LiDAR scan 데이터로부터
속도 명령(/cmd_vel)을 생성합니다.

Usage:
    ros2 run turtlebot3_policy jetson_inference_node
    
    # 또는 직접 실행:
    python3 src/deployment/jetson_inference_node.py
"""

import os
import sys
import math
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray

# TensorRT / ONNX Runtime
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class TurtleBotPolicyNode(Node):
    """
    RL 정책 추론 노드
    
    LiDAR scan과 목표 위치를 입력받아 CNN/MLP 정책으로
    선속도와 각속도를 출력합니다.
    
    ROS2 Interface:
        Subscribers:
            /scan (LaserScan) - LiDAR 스캔 데이터
            /odom (Odometry) - 오도메트리
            /goal_pose (PoseStamped) - 목표 위치 (선택)
        
        Publishers:
            /cmd_vel (Twist) - 속도 명령
            /policy/debug (Float32MultiArray) - 디버그 정보
    """
    
    def __init__(self):
        super().__init__('turtlebot_policy_node')
        
        # ========== 파라미터 선언 ==========
        self.declare_parameter('model_path', 'outputs/policy/turtlebot_policy.plan')
        self.declare_parameter('model_type', 'tensorrt')  # 'tensorrt' or 'onnx'
        self.declare_parameter('use_nav2_fallback', True)
        self.declare_parameter('policy_frequency', 10.0)    # Hz
        
        # ========== 설정 로드 ==========
        self.model_path = self.get_parameter('model_path').value
        self.model_type = self.get_parameter('model_type').value
        self.use_nav2_fallback = self.get_parameter('use_nav2_fallback').value
        self.policy_frequency = self.get_parameter('policy_frequency').value
        
        # ========== 정책 파라미터 ==========
        self.lidar_num_rays = 360
        self.lidar_downsampled = 36  # 10° 간격
        self.lidar_min_range = 0.15
        self.lidar_max_range = 3.5
        self.max_linear_vel = 0.22
        self.max_angular_vel = 2.84
        self.observation_dim = 39  # 36 (lidar) + 2 (goal) + 1 (heading error)
        
        # ========== 상태 ==========
        self.latest_scan = None
        self.latest_odom = None
        self.goal_position = None
        self.robot_position = (0.0, 0.0)
        self.robot_yaw = 0.0
        self.is_policy_running = False
        self.last_cmd_time = time.time()
        
        # ========== 정책 엔진 로드 ==========
        self.trt_engine = None
        self.trt_context = None
        self.ort_session = None
        
        self._load_policy()
        
        # ========== ROS2 통신 ==========
        # QoS 설정 (Best Effort, 🐢 속도)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(
            Float32MultiArray, '/policy/debug', 10
        )
        
        # 타이머 (정책 추론 주기)
        self.timer = self.create_timer(
            1.0 / self.policy_frequency, self.control_loop
        )
        
        # ========== 상태 플래그 ==========
        self.scan_received = False
        self.odom_received = False
        
        self.get_logger().info('TurtleBot Policy Node initialized')
        self.get_logger().info(f'  Model: {self.model_path}')
        self.get_logger().info(f'  Type: {self.model_type}')
        self.get_logger().info(f'  Frequency: {self.policy_frequency} Hz')
        self.get_logger().info(f'  Nav2 Fallback: {self.use_nav2_fallback}')
    
    def _load_policy(self):
        """TensorRT 또는 ONNX 모델 로드"""
        if not os.path.exists(self.model_path):
            self.get_logger().warn(
                f'Model not found: {self.model_path}. '
                f'Using heuristic fallback control.'
            )
            return
        
        try:
            if self.model_type == 'tensorrt' and TRT_AVAILABLE:
                self._load_tensorrt_engine()
            elif self.model_type == 'onnx' and ONNX_AVAILABLE:
                self._load_onnx_session()
            else:
                self.get_logger().warn(
                    f'Cannot load {self.model_type} model. '
                    f'TRT={TRT_AVAILABLE}, ONNX={ONNX_AVAILABLE}'
                )
        except Exception as e:
            self.get_logger().error(f'Failed to load policy: {e}')
    
    def _load_tensorrt_engine(self):
        """TensorRT 엔진 로드"""
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()
        
        self.get_logger().info(f'TensorRT engine loaded successfully')
        
        # 바인딩 정보
        for i in range(self.trt_engine.num_bindings):
            name = self.trt_engine.get_binding_name(i)
            dtype = self.trt_engine.get_binding_dtype(i)
            shape = self.trt_engine.get_binding_shape(i)
            self.get_logger().info(f'  Binding {i}: {name} {shape} {dtype}')
    
    def _load_onnx_session(self):
        """ONNX Runtime 세션 로드"""
        providers = [
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
        self.ort_session = ort.InferenceSession(
            self.model_path, providers=providers
        )
        
        # 입력/출력 정보
        for inp in self.ort_session.get_inputs():
            self.get_logger().info(f'  Input: {inp.name} {inp.shape} {inp.type}')
        for out in self.ort_session.get_outputs():
            self.get_logger().info(f'  Output: {out.name} {out.shape} {out.type}')
    
    def scan_callback(self, msg: LaserScan):
        """LiDAR scan 수신"""
        self.latest_scan = msg
        self.scan_received = True
    
    def odom_callback(self, msg: Odometry):
        """Odometry 수신"""
        self.latest_odom = msg
        self.odom_received = True
        
        # 위치 추출
        self.robot_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        
        # Yaw 추출 (Quaternion → Euler)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def goal_callback(self, msg: PoseStamped):
        """목표 위치 수신"""
        self.goal_position = (
            msg.pose.position.x,
            msg.pose.position.y,
        )
        self.get_logger().info(
            f'New goal: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})'
        )
    
    def preprocess_observation(self) -> np.ndarray:
        """
        센서 데이터를 정책 입력으로 전처리
        
        Returns:
            observation: (observation_dim,) numpy array
        """
        # 1. LiDAR 전처리 (360 → 36 downsampled)
        if self.latest_scan is not None:
            ranges = np.array(self.latest_scan.ranges)
            ranges = np.nan_to_num(ranges, nan=self.lidar_max_range)
            ranges = np.clip(ranges, self.lidar_min_range, self.lidar_max_range)
            ranges = ranges / self.lidar_max_range  # 정규화
            
            # 36개 빈으로 다운샘플 (각 빈 = 10°)
            lidar_downsampled = np.mean(
                ranges.reshape(36, 10), axis=1
            )
        else:
            lidar_downsampled = np.ones(36) * 0.5
        
        # 2. 목표 상대 위치
        if self.goal_position is not None:
            dx = self.goal_position[0] - self.robot_position[0]
            dy = self.goal_position[1] - self.robot_position[1]
        else:
            dx, dy = 1.0, 0.0  # 기본: 전방 1m
        
        # 거리 정규화 (8m 환경 기준)
        goal_relative = np.array([dx, dy]) / 8.0
        goal_relative = np.clip(goal_relative, -1.0, 1.0)
        
        # 3. Heading error
        goal_angle = math.atan2(dy, dx)
        heading_error = (goal_angle - self.robot_yaw + math.pi) % (2 * math.pi) - math.pi
        heading_error = np.array([heading_error / math.pi])  # [-1, 1] 정규화
        
        # 4. 관측값 결합
        observation = np.concatenate([
            lidar_downsampled,   # 36
            goal_relative,       # 2
            heading_error,       # 1
        ]).astype(np.float32)
        
        return observation
    
    def infer_action(self, observation: np.ndarray) -> np.ndarray:
        """
        정책 추론
        
        Args:
            observation: (39,) 관측값
        
        Returns:
            action: (2,) [linear_velocity, angular_velocity]
        """
        if self.trt_context is not None:
            # TensorRT 추론
            obs = np.expand_dims(observation, axis=0)  # (1, 39)
            
            # CUDA 메모리 할당 및 실행
            output = np.zeros((1, 2), dtype=np.float32)
            self.trt_context.execute_v2(
                bindings=[obs.ctypes.data, output.ctypes.data]
            )
            action = output[0]
            
        elif self.ort_session is not None:
            # ONNX 추론
            obs = np.expand_dims(observation, axis=0).astype(np.float32)
            output = self.ort_session.run(
                ['action'], {'observation': obs}
            )
            action = output[0][0]
            
        else:
            # Fallback: 간단한 충돌 회피
            action = self._heuristic_control(observation)
        
        return action
    
    def _heuristic_control(self, observation: np.ndarray) -> np.ndarray:
        """
        Heuristic 충돌 회피 (Policy 로드 실패 시 Fallback)
        
        간단한 Potential Field 기반 제어:
        - 전방 장애물 → 회전
        - 목표 방향 → 진행
        """
        lidar = observation[:36]
        goal_dx, goal_dy = observation[36:38]
        heading_err = observation[38]
        
        # 전방 장애물 감지 (중앙 60°)
        front_indices = list(range(15, 21))  # 150° ~ 210° (로봇 기준 전방)
        front_dist = np.min(lidar[front_indices]) * self.lidar_max_range
        
        linear_vel = 0.0
        angular_vel = 0.0
        
        if front_dist < 0.3:
            # 충돌 직전 → 회피
            # 왼쪽/오른쪽 중 더 넓은 쪽으로 회전
            left_min = np.min(lidar[8:15])  # 80° ~ 150°
            right_min = np.min(lidar[21:28])  # 210° ~ 280°
            
            linear_vel = -0.05  # 약간 후진
            angular_vel = 0.5 if left_min > right_min else -0.5
        else:
            # 목표 방향으로 주행
            linear_vel = min(0.15, front_dist * 0.5)
            angular_vel = max(-1.0, min(1.0, heading_err * 2.0))
        
        return np.array([linear_vel, angular_vel])
    
    def control_loop(self):
        """정책 추론 루프 (Timer callback)"""
        if not self.scan_received or not self.odom_received:
            return
        
        # 관측값 전처리
        observation = self.preprocess_observation()
        
        # 정책 추론
        action = self.infer_action(observation)
        
        # 속도 스케일링
        linear_vel = float(action[0]) * self.max_linear_vel
        angular_vel = float(action[1]) * self.max_angular_vel
        
        # 속도 제한
        linear_vel = np.clip(linear_vel, -0.1, self.max_linear_vel)
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        # Twist 메시지 발행
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)
        
        # 디버그 정보 발행
        debug_msg = Float32MultiArray()
        debug_msg.data = [
            linear_vel,                     # 0: linear velocity
            angular_vel,                    # 1: angular velocity
            float(time.time()),             # 2: timestamp
        ]
        self.debug_pub.publish(debug_msg)
        
        self.is_policy_running = True
        self.last_cmd_time = time.time()


def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotPolicyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
        # 정지 명령
        twist = Twist()
        node.cmd_vel_pub.publish(twist)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 7.4 TurtleBot3 모터 드라이버 노드

**`src/deployment/turtlebot_driver.py`**:

```python
#!/usr/bin/env python3
"""
TurtleBot3 Hardware Driver for Jetson Orin Nano

OpenCR 보드와 UART 통신을 통해 TurtleBot3의 모터를 제어합니다.

Usage:
    ros2 run turtlebot3_driver turtlebot_driver
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import struct
import time
import threading


class TurtleBotDriver(Node):
    """
    TurtleBot3 하드웨어 드라이버
    
    /cmd_vel Twist 메시지를 수신하여 OpenCR 보드의 
    DYNAMIXEL 모터 제어 명령으로 변환합니다.
    """
    
    def __init__(self):
        super().__init__('turtlebot_driver')
        
        # 파라미터
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('wheel_separation', 0.160)
        self.declare_parameter('wheel_radius', 0.033)
        
        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        
        # 시리얼 통신 (OpenCR)
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=0.1
            )
            self.get_logger().info(f'Connected to OpenCR on {port}')
        except serial.SerialException as e:
            self.get_logger().error(f'Cannot open {port}: {e}')
            self.ser = None
        
        # Subscriber
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # 타이머 (Watchdog: 0.5초 이상 명령 없으면 정지)
        self.last_cmd_time = time.time()
        self.timer = self.create_timer(0.1, self.watchdog)
        
        # 모터 상태
        self.target_linear = 0.0
        self.target_angular = 0.0
        
        self.get_logger().info('TurtleBot Driver initialized')
    
    def cmd_callback(self, msg: Twist):
        """속도 명령 수신"""
        self.target_linear = msg.linear.x
        self.target_angular = msg.angular.z
        self.last_cmd_time = time.time()
        
        # OpenCR로 명령 전송
        self._send_motor_command(self.target_linear, self.target_angular)
    
    def _send_motor_command(self, linear_vel: float, angular_vel: float):
        """
        차동 구동 속도를 좌우 바퀴 속도로 변환하여 OpenCR로 전송
        
        Args:
            linear_vel: 선속도 (m/s)
            angular_vel: 각속도 (rad/s)
        """
        if self.ser is None:
            return
        
        wheel_sep = self.get_parameter('wheel_separation').value
        wheel_rad = self.get_parameter('wheel_radius').value
        
        # 역기구학: 차동 구동 → 휠 속도
        left_wheel_speed = (linear_vel - angular_vel * wheel_sep / 2.0) / wheel_rad
        right_wheel_speed = (linear_vel + angular_vel * wheel_sep / 2.0) / wheel_rad
        
        # RPM 변환 (rad/s → RPM)
        left_rpm = left_wheel_speed * 60.0 / (2.0 * math.pi)
        right_rpm = right_wheel_speed * 60.0 / (2.0 * math.pi)
        
        # OpenCR 프로토콜로 전송
        # (실제 프로토콜은 ROBOTIS DYNAMIXEL SDK 참조)
        try:
            # 간소화된 프로토콜 예시
            packet = bytearray()
            packet.append(0xFF)  # Header
            packet.append(0xFF)
            packet.append(0xFD)  # Protocol 2.0
            packet.extend(struct.pack('<I', int(left_rpm * 100)))   # 왼쪽 RPM * 100
            packet.extend(struct.pack('<I', int(right_rpm * 100)))  # 오른쪽 RPM * 100
            packet.append(0x0D)  # Footer
            packet.append(0x0A)
            
            self.ser.write(packet)
        except Exception as e:
            self.get_logger().error(f'Serial write error: {e}')
    
    def watchdog(self):
        """Watchdog: 명령이 일정 시간 없으면 정지"""
        if time.time() - self.last_cmd_time > 0.5:
            self._send_motor_command(0.0, 0.0)


def main(args=None):
    import math  # watchdog callback에서 사용
    rclpy.init(args=args)
    node = TurtleBotDriver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 7.5 실행 워크플로우

### 7.5.1 Jetson Orin Nano에서 실행

```bash
# ============================================
# Terminal 1: ROS2 Core + TurtleBot3 기본
# ============================================
source /opt/ros/humble/setup.bash
source ~/tb3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=42

# TurtleBot3 기본 구동 (OpenCR)
ros2 run turtlebot3_bringup turtlebot3_robot

# 또는 모터 드라이버 직접 실행:
python3 ~/policy_ws/src/deployment/turtlebot_driver.py


# ============================================
# Terminal 2: Policy Inference Node
# ============================================
source ~/policy_ws/install/setup.bash
export ROS_DOMAIN_ID=42

# TensorRT 기반 정책 추론
python3 src/deployment/jetson_inference_node.py \
    --model_path outputs/policy/turtlebot_policy.plan


# ============================================
# Terminal 3: RViz 모니터링 (데스크톱에서)
# ============================================
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
rviz2
```

### 7.5.2 전체 시스템 실행 스크립트

**`scripts/run_jetson.sh`**:

```bash
#!/bin/bash
# Jetson Orin Nano 전체 실행 스크립트
# Usage: bash scripts/run_jetson.sh

set -e

# ========== 설정 ==========
ROS_DOMAIN_ID=42
TURTLEBOT3_MODEL=burger
POLICY_MODEL="outputs/policy/turtlebot_policy.plan"
USE_NAV2=false

# ========== ROS2 환경 ==========
source /opt/ros/humble/setup.bash
source ~/tb3_ws/install/setup.bash
export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
export TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo "=============================================="
echo "TurtleBot3 Autonomous Navigation"
echo "=============================================="
echo "  Policy Model: $POLICY_MODEL"
echo "  Domain ID:    $ROS_DOMAIN_ID"
echo "  Robot Model:  $TURTLEBOT3_MODEL"
echo "  Nav2:         $USE_NAV2"
echo "=============================================="

# ========== 센서 확인 ==========
echo "[Check] LiDAR..."
if [ -c /dev/ttyUSB0 ]; then
    echo "  LiDAR found at /dev/ttyUSB0"
else
    echo "  ⚠️  LiDAR not found at /dev/ttyUSB0"
fi

echo "[Check] OpenCR..."
if [ -c /dev/ttyACM0 ]; then
    echo "  OpenCR found at /dev/ttyACM0"
else
    echo "  ⚠️  OpenCR not found at /dev/ttyACM0"
fi

# ========== 실행 ==========
echo ""
echo "[Launch] Starting robots..."

# TurtleBot3 Bringup (백그라운드)
ros2 launch turtlebot3_bringup turtlebot3_robot.launch.py &
TB3_PID=$!
sleep 5

# Policy Inference Node (백그라운드)
if [ -f "$POLICY_MODEL" ]; then
    echo "[Launch] Policy inference node..."
    python3 ~/policy_ws/src/deployment/jetson_inference_node.py \
        --model_path $POLICY_MODEL &
    POLICY_PID=$!
    sleep 2
else
    echo "[WARN] Policy model not found. Using Nav2 fallback..."
    if [ "$USE_NAV2" = true ]; then
        ros2 launch nav2_bringup navigation_launch.py &
        NAV2_PID=$!
    fi
fi

# ========== 모니터링 ==========
echo ""
echo "=============================================="
echo "All systems running!"
echo "  TurtleBot3: PID $TB3_PID"
echo "  Policy:     PID $POLICY_PID"
echo "=============================================="
echo ""
echo "Topics:"
echo "  /scan       - LiDAR data"
echo "  /odom       - Odometry"
echo "  /cmd_vel    - Velocity commands"
echo "  /policy/debug - Policy debug info"
echo ""
echo "Press Ctrl+C to stop all processes"

# ========== Cleanup on exit ==========
cleanup() {
    echo ""
    echo "[Shutdown] Stopping all processes..."
    kill $POLICY_PID 2>/dev/null
    kill $NAV2_PID 2>/dev/null
    kill $TB3_PID 2>/dev/null
    wait
    echo "[Shutdown] Complete."
}

trap cleanup EXIT INT TERM

# Wait for any background process to exit
wait
```

## 7.6 성능 최적화

| 최적화 | 방법 | 기대 효과 |
|--------|------|-----------|
| TensorRT FP16 | `--fp16` 변환 | 추론 속도 2배, VRAM 50% 감소 |
| INT8 Calibration | 추가 캘리브레이션 | 추론 속도 3배 (선택 사항) |
| Batch Size 1 | 단일 추론 최적화 | Jetson에 최적 |
| LiDAR Downsampling | 360 → 36 rays | 전처리 속도 10배 |
| CUDA Stream | 비동기 추론 | CPU/GPU 동시 활용 |
| Jetson MAXN | 25W 모드 | 최대 추론 성능 |
| CPU Affinity | IRQ 할당 | 실시간성 향상 |

### 예상 성능 (Jetson Orin Nano)

| 구성 | 추론 시간 | FPS | 전력 |
|------|-----------|-----|------|
| ONNX (FP32) | ~15ms | ~66Hz | 15W |
| TensorRT (FP16) | ~5ms | ~200Hz | 15W |
| TensorRT (INT8) | ~3ms | ~333Hz | 15W |
| TensorRT FP16 + MAXN | ~3ms | ~333Hz | 25W |

> 정책 추론은 10Hz(100ms)면 충분하므로 모든 구성이 실시간 요구사항을 만족합니다.

## 7.7 문제 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| `/dev/ttyACM0` 없음 | OpenCR 미연결 | `ls /dev/tty*` 확인, udev 규칙 설정 |
| LiDAR 안 켜짐 | USB 전원 부족 | 허브 전원 공급 확인 |
| 정책 출력 0만 나옴 | 잘못된 모델 경로 | `--model_path` 절대 경로 확인 |
| 추론 속도 느림 | GPU 미사용 | `onnxruntime-gpu` 설치 확인 |
| ROS2 통신 안 됨 | DDS 불일치 | `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION` 확인 |
| 충돌 발생 | 정책 오류 | Nav2 fallback 활성화 `--use_nav2_fallback` |
