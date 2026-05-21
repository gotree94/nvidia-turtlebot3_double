# 08. Dual TurtleBot3 시스템 확장

> **Raspberry Pi 5 + Jetson Orin Nano 이기종 듀얼 로봇 시스템**

## 8.1 시스템 개요

```
                    ┌─────────────────────────────────────┐
                    │         ROS2 DDS Network             │
                    │        (CycloneDDS / FastDDS)        │
                    │         Domain ID: 42               │
                    └─────────────────────────────────────┘
                              ▲              ▲
                              │              │
                ┌─────────────┴──┐    ┌──────┴─────────────┐
                │  Robot #1      │    │  Robot #2          │
                │  (Jetson Orin) │    │  (Raspberry Pi 5)  │
                │                │    │                    │
                │  역할: 리더    │    │  역할: 팔로워     │
                │  • 고성능 정책 │    │  • 경량 Nav2      │
                │  • TensorRT    │    │  • 원격 제어       │
                │  • 카메라 처리 │    │  • SLAM 경량화    │
                │  • 멀티모달    │    │                    │
                └────────────────┘    └────────────────────┘
```

### 로봇별 역할 분담

| 기능 | Robot 1 (Jetson Orin Nano) | Robot 2 (Raspberry Pi 5) |
|------|---------------------------|--------------------------|
| **정책 추론** | TensorRT 고속 추론 (333Hz) | Nav2 기반 경로 계획 |
| **네비게이션** | RL Policy + Nav2 Hybrid | Nav2 Standalone |
| **카메라** | USB Camera / RealSense | Pi Camera v3 |
| **SLAM** | Full SLAM Toolbox | 경량 SLAM (실시간 맵 공유) |
| **통신** | Gigabit Ethernet | Wi-Fi 6 |
| **전력** | 25W (MAXN) | 5W |
| **ROS2** | Humble | Jazzy |

## 8.2 네임스페이스 설정

ROS2 네임스페이스를 통해 두 로봇을 분리합니다.

### 8.2.1 네임스페이스 설계

```
/tb1/                        ← Robot 1 (Jetson)
├── cmd_vel
├── odom
├── scan
├── tf
├── map
├── policy/debug
└── goal_pose

/tb2/                        ← Robot 2 (RPi)
├── cmd_vel
├── odom
├── scan
├── tf
├── map
└── goal_pose

/ (Global)                   ← 공통 토픽
├── /map                     # 글로벌 맵 (공유)
├── /tf_static               # 정적 TF
├── /robot_formation         # 대형 제어 명령
└── /multi_robot_status      # 상태 동기화
```

### 8.2.2 Robot 1 런치 (Jetson Orin Nano)

```python
# src/deployment/launch_robot1.py
import launch
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction


def generate_launch_description():
    namespace = 'tb1'
    
    return launch.LaunchDescription([
        GroupAction(
            actions=[
                PushRosNamespace(namespace),
                
                # TurtleBot3 하드웨어
                Node(
                    package='turtlebot3_bringup',
                    executable='turtlebot3_robot',
                    name='turtlebot3_robot',
                    namespace=namespace,
                ),
                
                # RL Policy 추론
                Node(
                    package='turtlebot3_policy',
                    executable='jetson_inference_node',
                    name='policy_node',
                    namespace=namespace,
                    parameters=[{
                        'model_path': 'outputs/policy/turtlebot_policy.plan',
                        'model_type': 'tensorrt',
                        'policy_frequency': 10.0,
                    }],
                ),
                
                # SLAM
                Node(
                    package='slam_toolbox',
                    executable='async_slam_toolbox_node',
                    name='slam_toolbox',
                    namespace=namespace,
                    parameters=[{
                        'use_sim_time': False,
                        'map_frame': f'{namespace}/map',
                        'odom_frame': f'{namespace}/odom',
                        'base_frame': f'{namespace}/base_link',
                    }],
                ),
                
                # Odometry → TF 브로드캐스터
                Node(
                    package='robot_localization',
                    executable='ekf_node',
                    name='ekf_filter_node',
                    namespace=namespace,
                    parameters=['config/ekf_params.yaml'],
                ),
            ]
        ),
        
        # 글로벌 토픽 브릿지
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='global_tf_bridge',
            arguments=['0', '0', '0', '0', '0', '0', 'map', f'{namespace}/map'],
        ),
    ])
```

### 8.2.3 Robot 2 런치 (Raspberry Pi 5)

```python
# src/deployment/launch_robot2.py
import launch
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction


def generate_launch_description():
    namespace = 'tb2'
    
    return launch.LaunchDescription([
        GroupAction(
            actions=[
                PushRosNamespace(namespace),
                
                # TurtleBot3 하드웨어 (UART 경로 다름)
                Node(
                    package='turtlebot3_bringup',
                    executable='turtlebot3_robot',
                    name='turtlebot3_robot',
                    namespace=namespace,
                    parameters=[{'port': '/dev/ttyAMA0'}],
                ),
                
                # Nav2 경량 설정
                Node(
                    package='nav2_controller',
                    executable='controller_server',
                    name='controller_server',
                    namespace=namespace,
                    parameters=['config/nav2_light_params.yaml'],
                ),
                Node(
                    package='nav2_planner',
                    executable='planner_server',
                    name='planner_server',
                    namespace=namespace,
                    parameters=['config/nav2_light_params.yaml'],
                ),
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager',
                    namespace=namespace,
                    parameters=[{
                        'autostart': True,
                        'node_names': ['controller_server', 'planner_server'],
                    }],
                ),
            ]
        ),
    ])
```

## 8.3 대형 제어 (Formation Control)

**`src/deployment/formation_controller.py`**:

```python
#!/usr/bin/env python3
"""
Dual TurtleBot3 Formation Controller

두 로봇의 대형(Formation)을 제어합니다.
Leader-Follower 방식으로 Robot 1이 선두에서 정책 추론,
Robot 2가 뒤따르는 형태입니다.

Usage:
    ros2 run turtlebot3_multi formation_controller --formation column
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
import math
import numpy as np


class FormationController(Node):
    """
    듀얼 로봇 대형 제어
    
    Formation Types:
        - column: 일렬 종대 (Robot 1 앞, Robot 2 뒤)
        - line: 횡대 (나란히)
        - diamond: 다이아몬드 (45° 대각선)
        - staggered: 지그재그
    """
    
    def __init__(self):
        super().__init__('formation_controller')
        
        # 파라미터
        self.declare_parameter('formation', 'column')
        self.declare_parameter('separation', 0.8)  # 로봇 간 거리 (m)
        self.declare_parameter('leader', 'tb1')
        self.declare_parameter('follower', 'tb2')
        
        self.formation = self.get_parameter('formation').value
        self.separation = self.get_parameter('separation').value
        self.leader = self.get_parameter('leader').value
        self.follower = self.get_parameter('follower').value
        
        # TF 버퍼
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Leader 상태
        self.leader_pose = None
        self.leader_twist = None
        
        # Follower 제어 명령 발행
        self.follower_cmd_pub = self.create_publisher(
            Twist, f'/{self.follower}/cmd_vel', 10
        )
        
        # Leader 구독
        self.create_subscription(
            Odometry, f'/{self.leader}/odom', self.leader_odom_callback, 10
        )
        
        # 제어 타이머 (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info(
            f'Formation Controller initialized: {self.formation} mode'
        )
        self.get_logger().info(f'  Leader: {self.leader}, Follower: {self.follower}')
        self.get_logger().info(f'  Separation: {self.separation}m')
    
    def leader_odom_callback(self, msg: Odometry):
        """Leader 위치 업데이트"""
        self.leader_pose = msg.pose.pose
        self.leader_twist = msg.twist.twist
    
    def get_follower_pose(self):
        """TF로 Follower 위치 조회"""
        try:
            trans = self.tf_buffer.lookup_transform(
                'map',
                f'{self.follower}/base_link',
                rclpy.time.Time()
            )
            return trans.transform.translation, trans.transform.rotation
        except Exception as e:
            return None, None
    
    def compute_follower_target(self) -> tuple:
        """
        대형에 따른 Follower 목표 위치 계산
        
        Returns:
            (target_x, target_y, target_yaw) in map frame
        """
        if self.leader_pose is None:
            return None
        
        # Leader의 현재 위치와 방향
        lx = self.leader_pose.position.x
        ly = self.leader_pose.position.y
        
        q = self.leader_pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        lyaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 대형별 오프셋 계산
        if self.formation == 'column':
            # 일렬 종대: 뒤에 위치
            dx = -self.separation * math.cos(lyaw)
            dy = -self.separation * math.sin(lyaw)
        elif self.formation == 'line':
            # 횡대: 옆에 위치
            dx = -self.separation * math.sin(lyaw)
            dy = self.separation * math.cos(lyaw)
        elif self.formation == 'diamond':
            # 다이아몬드: 45° 대각선 뒤
            dx = -self.separation * 0.707 * math.cos(lyaw) - self.separation * 0.707 * math.sin(lyaw)
            dy = -self.separation * 0.707 * math.sin(lyaw) + self.separation * 0.707 * math.cos(lyaw)
        elif self.formation == 'staggered':
            # 지그재그
            dx = -self.separation * math.cos(lyaw)
            dy = -0.3 * math.sin(lyaw * 2)  # 사인파 오프셋
        else:
            dx = -self.separation * math.cos(lyaw)
            dy = -self.separation * math.sin(lyaw)
        
        return (lx + dx, ly + dy, lyaw)
    
    def control_loop(self):
        """대형 제어 루프"""
        target = self.compute_follower_target()
        if target is None:
            return
        
        target_x, target_y, target_yaw = target
        
        # Follower 현재 위치
        f_pos, f_rot = self.get_follower_pose()
        if f_pos is None:
            return
        
        # 오차 계산
        dx = target_x - f_pos.x
        dy = target_y - f_pos.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Follower yaw
        q = f_rot
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        fyaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 간단한 P 제어
        angle_to_target = math.atan2(dy, dx)
        angle_error = (angle_to_target - fyaw + math.pi) % (2 * math.pi) - math.pi
        
        linear_vel = min(0.15, distance * 0.5)  # 속도 제한
        angular_vel = max(-0.5, min(0.5, angle_error * 1.0))
        
        # 목표 방향 정렬 보정
        if abs(angle_error) > 0.5:
            linear_vel *= 0.3  # 방향 정렬 중에는 속도 감소
        
        # Twist 명령 발행
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.follower_cmd_pub.publish(twist)
        
        # 디버그 로그
        self.get_logger().debug(
            f'Target: ({target_x:.2f}, {target_y:.2f}) | '
            f'Error: {distance:.2f}m, {math.degrees(angle_error):.1f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FormationController()
    
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

## 8.4 멀티 로봇 매니저

**`src/deployment/multi_robot_manager.py`**:

```python
#!/usr/bin/env python3
"""
멀티 로봇 시스템 관리자

두 로봇의 상태 모니터링, 글로벌 목표 할당, 충돌 회피를 담당합니다.

Usage:
    ros2 run turtlebot3_multi multi_robot_manager
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
import json


class MultiRobotManager(Node):
    """멀티 로봇 중앙 관리 노드"""
    
    def __init__(self):
        super().__init__('multi_robot_manager')
        
        # 로봇 상태
        self.robots = {
            'tb1': {'odom': None, 'status': 'idle', 'battery': 100},
            'tb2': {'odom': None, 'status': 'idle', 'battery': 100},
        }
        
        # Subscribers
        for robot_id in self.robots:
            self.create_subscription(
                Odometry, f'/{robot_id}/odom',
                lambda msg, rid=robot_id: self.odom_callback(msg, rid),
                10
            )
            self.create_subscription(
                String, f'/{robot_id}/status',
                lambda msg, rid=robot_id: self.status_callback(msg, rid),
                10
            )
        
        # Publishers
        self.global_goal_pub = self.create_publisher(PoseStamped, '/global_goal', 10)
        self.robot_markers_pub = self.create_publisher(MarkerArray, '/robot_markers', 10)
        
        # 타이머
        self.timer = self.create_timer(1.0, self.status_check)
        self.marker_timer = self.create_timer(0.5, self.publish_markers)
        
        self.get_logger().info('Multi Robot Manager initialized')
    
    def odom_callback(self, msg: Odometry, robot_id: str):
        self.robots[robot_id]['odom'] = msg
    
    def status_callback(self, msg: String, robot_id: str):
        try:
            status = json.loads(msg.data)
            self.robots[robot_id].update(status)
        except:
            self.robots[robot_id]['status'] = msg.data
    
    def status_check(self):
        """로봇 상태 정기 점검"""
        active_count = sum(
            1 for r in self.robots.values() 
            if r['odom'] is not None
        )
        self.get_logger().info(
            f'Robots active: {active_count}/{len(self.robots)}'
        )
        
        if active_count < len(self.robots):
            inactive = [
                rid for rid, r in self.robots.items() 
                if r['odom'] is None
            ]
            self.get_logger().warn(f'Inactive: {", ".join(inactive)}')
    
    def publish_markers(self):
        """RViz용 로봇 마커 발행"""
        marker_array = MarkerArray()
        
        for i, (robot_id, state) in enumerate(self.robots.items()):
            if state['odom'] is None:
                continue
            
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = robot_id
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose = state['odom'].pose.pose
            marker.scale.x = 0.14
            marker.scale.y = 0.14
            marker.scale.z = 0.10
            
            # 로봇별 색상
            marker.color.r = 0.2 if i == 0 else 0.8
            marker.color.g = 0.8 if i == 0 else 0.2
            marker.color.b = 0.2
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        self.robot_markers_pub.publish(marker_array)
    
    def set_global_goal(self, x: float, y: float, frame_id: str = 'map'):
        """글로벌 목표 설정"""
        goal = PoseStamped()
        goal.header.frame_id = frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0
        
        self.global_goal_pub.publish(goal)
        self.get_logger().info(f'Global goal set: ({x:.2f}, {y:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotManager()
    
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

## 8.5 듀얼 로봇 실행 가이드

### 8.5.1 사전 요구사항

| 항목 | Robot 1 (Jetson) | Robot 2 (RPi 5) |
|------|-----------------|-----------------|
| OS | Ubuntu 22.04 (JetPack 6) | Ubuntu 24.04 LTS |
| ROS2 | Humble | Jazzy |
| TurtleBot3 | OpenCR + LDS-01 | OpenCR + LDS-01 |
| 네트워크 | Ethernet (고정 IP: 10.0.0.20) | Wi-Fi 6 (고정 IP: 10.0.0.30) |
| 전원 | 25W 어댑터 + 보조 배터리 | 5V/3A 어댑터 |

### 8.5.2 ROS2 네트워크 설정

**모든 로봇의 `/etc/sysctl.d/99-ros-network.conf`**:
```conf
# ROS2 DDS Multicast 설정
net.core.rmem_max=2147483647
net.core.wmem_max=2147483647
net.ipv4.ipfrag_time=3
net.ipv4.ipfrag_high_thresh=134217728
```

**CycloneDDS XML 설정 (`config/cyclonedds.xml`)**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain Id="42">
        <General>
            <Interfaces>
                <!-- Wi-Fi 인터페이스 (RPi) -->
                <NetworkInterface name="wlan0" priority="default"/>
                <!-- Ethernet 인터페이스 (Jetson) -->
                <NetworkInterface name="eth0" priority="default"/>
            </Interfaces>
            <AllowMulticast>true</AllowMulticast>
            <MaxMessageSize>65500B</MaxMessageSize>
            <FragmentSize>1280B</FragmentSize>
        </General>
        <Discovery>
            <ParticipantIndex>auto</ParticipantIndex>
            <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
        </Discovery>
        <Internal>
            <Watermarks>
                <WhcHigh>500kB</WhcHigh>
            </Watermarks>
        </Internal>
    </Domain>
</CycloneDDS>
```

### 8.5.3 듀얼 로봇 실행 순서

```bash
# ============================================
# Step 1: Robot 2 (RPi 5) - Follower
# ============================================
ssh ubuntu@10.0.0.30

# ROS2 환경 설정
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/ubuntu/config/cyclonedds.xml
export TURTLEBOT3_MODEL=burger

# TurtleBot3 bringup + Nav2
ros2 launch turtlebot3_bringup turtlebot3_robot.launch.py \
    namespace:=tb2 &

# Nav2 경량 실행
ros2 launch nav2_bringup navigation_launch.py \
    namespace:=tb2 \
    params_file:=/home/ubuntu/config/nav2_light_params.yaml


# ============================================
# Step 2: Robot 1 (Jetson Orin Nano) - Leader
# ============================================
ssh nvidia@10.0.0.20

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export TURTLEBOT3_MODEL=burger

# TurtleBot3 bringup
ros2 launch turtlebot3_bringup turtlebot3_robot.launch.py \
    namespace:=tb1 &

# RL Policy 추론
python3 src/deployment/jetson_inference_node.py \
    --model_path outputs/policy/turtlebot_policy.plan &

# Formation Controller
python3 src/deployment/formation_controller.py \
    --formation column --separation 0.8 &


# ============================================
# Step 3: Monitor (Desktop)
# ============================================
export ROS_DOMAIN_ID=42

# Multi Robot Manager
python3 src/deployment/multi_robot_manager.py &

# RViz
rviz2 -d config/multi_robot.rviz
```

### 8.5.4 듀얼 로봇 실행 스크립트

**`scripts/run_dual_robot.sh`**:

```bash
#!/bin/bash
# 듀얼 TurtleBot3 실행 스크립트 (Desktop에서 원격 실행)
# Usage: bash scripts/run_dual_robot.sh

set -e

JETSON_IP="10.0.0.20"
RPI_IP="10.0.0.30"
ROS_DOMAIN_ID=42

echo "=============================================="
echo "Dual TurtleBot3 System Launch"
echo "=============================================="

# Step 1: RPi 5 (Follower) 실행
echo "[1/3] Starting Robot 2 (RPi 5)..."
ssh ubuntu@$RPI_IP "bash -s" << 'EOF'
    export ROS_DOMAIN_ID=42
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export TURTLEBOT3_MODEL=burger
    
    # 기존 프로세스 정리
    pkill -f turtlebot3_robot 2>/dev/null || true
    pkill -f nav2 2>/dev/null || true
    
    # TurtleBot3 + Nav2 실행
    source /opt/ros/jazzy/setup.bash
    source ~/tb3_ws/install/setup.bash
    
    ros2 launch ~/policy_ws/src/deployment/launch_robot2.py &
    echo "Robot 2 started"
EOF
sleep 5

# Step 2: Jetson Orin Nano (Leader) 실행
echo "[2/3] Starting Robot 1 (Jetson Orin Nano)..."
ssh nvidia@$JETSON_IP "bash -s" << 'EOF'
    export ROS_DOMAIN_ID=42
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export TURTLEBOT3_MODEL=burger
    
    pkill -f turtlebot3_robot 2>/dev/null || true
    pkill -f jetson_inference 2>/dev/null || true
    
    source /opt/ros/humble/setup.bash
    source ~/tb3_ws/install/setup.bash
    
    ros2 launch ~/policy_ws/src/deployment/launch_robot1.py &
    sleep 3
    
    # Formation Controller
    python3 ~/policy_ws/src/deployment/formation_controller.py \
        --formation column --separation 0.8 &
    
    echo "Robot 1 started"
EOF
sleep 3

# Step 3: Desktop 모니터링
echo "[3/3] Starting monitoring..."
export ROS_DOMAIN_ID=42

source /opt/ros/humble/setup.bash
python3 src/deployment/multi_robot_manager.py &
sleep 2

echo ""
echo "=============================================="
echo "Dual robot system is running!"
echo "  Robot 1 (Leader):  $JETSON_IP - RL Policy"
echo "  Robot 2 (Follower): $RPI_IP - Nav2"
echo "=============================================="
echo ""
echo "To send a formation command:"
echo "  ros2 topic pub /robot_formation std_msgs/String '{data: column}'"
echo ""
echo "Press Ctrl+C to stop all"
wait
```

## 8.6 실험 시나리오

### 시나리오 1: Leader-Follower Column
```
Robot 1 (Jetson)          Robot 2 (RPi)
    ●                          ●
    ▲                          ▲
    │                          │
    │    0.8m separation       │
    └──────────────────────────┘
    Leader가 경로 생성 → Follower가 추종
```

### 시나리오 2: Side-by-Side Exploration
```
      Robot 1           Robot 2
    ●──────────────●
    │  0.8m width  │
    └──────────────┘
    병렬 탐사로 더 넓은 영역 커버
```

### 시나리오 3: Staggered Formation (충돌 회피)
```
    Robot 1
    ●
        ●
    Robot 2
    지그재그 대형으로 좁은 공간 통과
```

### 시나리오 4: Cooperative Transportation
```
    ┌──────────────────────┐
    │        Object        │
    └──────────────────────┘
    ●──────────────────────●
    Robot 1           Robot 2
    함께 물체 운반 (가상)
```

## 8.7 문제 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| RPi에서 ROS2 토픽 안 보임 | Domain ID 불일치 | `ROS_DOMAIN_ID` 일치 확인 |
| 대형 유지 실패 | WiFi 지연 | Ethernet 사용 권장 |
| Follower가 Leader 놓침 | 속도 차이 | formation 속도 제한 (max_linear_vel 0.15) |
| SLAM 맵 불일치 | TF 불일치 | 네임스페이스 TF 설정 확인 |
| Jetson 전력 부족 | MAXN 모드 확인 | `sudo nvpmodel -m 0` 실행 |
