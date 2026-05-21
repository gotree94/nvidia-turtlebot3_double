# 04. Isaac Sim 시뮬레이션 환경

> **TurtleBot3 시뮬레이션 환경 구축, ROS2 브릿지, 센서 설정**

## 4.1 Isaac Sim 개요

NVIDIA Isaac Sim은 Omniverse 플랫폼 위에 구축된 물리 기반 로봇 시뮬레이터입니다.

### 주요 기능
- **PhysX 5**: GPU 가속 물리 시뮬레이션
- **MDL (Material Definition Language)**: 사실적인 머티리얼
- **RTX Renderer**: 실시간 레이 트레이싱
- **ROS2 Bridge**: ROS2와의 완벽한 양방향 통합
- **Replicator**: 합성 데이터 생성 (Cosmos 연동)
- **OmniGraph**: 비주얼 노드 기반 프로그래밍

### 버전 호환성

| Isaac Sim | ROS2 버전 | Python | CUDA | 특징 |
|-----------|-----------|--------|------|------|
| 2025.2 | Humble / Jazzy | 3.10 | 12.4 | 최신 기능, Isaac Lab 2.1+ |
| 2025.1 | Humble / Jazzy | 3.10 | 12.4 | 안정 버전 |
| 2024.2 | Humble | 3.10 | 12.2 | LTS |

## 4.2 Isaac Sim TurtleBot3 워크플로우

### 4.2.1 전체 흐름

```
1. URDF 준비 ──URDF Importer──► 2. USD 변환
                                       │
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                   3a. 환경 배치   3b. 센서 설정  3c. 컨트롤러 구성
                          │            │            │
                          └────────────┼────────────┘
                                       ▼
                             4. ROS2 Bridge 설정
                                       ▼
                             5. 시뮬레이션 실행
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                    6a. 수동 주행             6b. Nav2 연동
                        (teleop)               (자율 주행)
                                       │
                                       ▼
                             7. Isaac Lab 학습
```

### 4.2.2 시뮬레이션 설정 스크립트

**`src/isaac_sim/setup_simulation.py`**:

```python
"""
TurtleBot3 Isaac Sim 시뮬레이션 환경 설정 스크립트

이 스크립트는 Isaac Sim 내에서 Python Script Editor 또는
./python.sh로 실행합니다.

Usage:
    /isaac-sim/python.sh /workspace/src/isaac_sim/setup_simulation.py
"""

import os
import sys
import math
import numpy as np
import carb
import omni
import omni.usd
import omni.graph.core as og
from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics
from omni.isaac.core import SimulationContext
from omni.isaac.core.world import World
from omni.isaac.core.prims import XFormPrim, GeometryPrim
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.sensor import LidarRtxSensor


class TurtleBotSimulation:
    """TurtleBot3 시뮬레이션 환경 클래스"""
    
    def __init__(self, usd_path: str, headless: bool = False):
        self.usd_path = usd_path
        self.headless = headless
        self.world = None
        self.robot = None
        self.lidar = None
        
    def setup_world(self):
        """기본 월드 환경 설정"""
        print("[Setup] Creating simulation world...")
        
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # 조명 설정
        stage = get_current_stage()
        prim = stage.GetPrimAtPath("/World")
        
        # 환경 라이트
        from pxr import UsdLux
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome_light.CreateIntensityAttr(1000)
        
        # 구역 라이트
        rect_light = UsdLux.RectLight.Define(stage, Sdf.Path("/World/RectLight"))
        rect_light.CreateIntensityAttr(500)
        rect_light.AddTranslateOp().Set(Gf.Vec3f(2, 2, 3))
        rect_light.AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 0))
        
        print("[Setup] World ready.")
        
    def load_robot(self, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)):
        """TurtleBot3 USD 로드"""
        print(f"[Robot] Loading TurtleBot3 from: {self.usd_path}")
        
        if not os.path.exists(self.usd_path):
            print(f"[ERROR] USD file not found: {self.usd_path}")
            # URDF에서 변환하지 않은 경우 기본 Jetbot 사용
            assets_root = get_assets_root_path()
            if assets_root:
                jetbot_path = f"{assets_root}/Isaac/Robots/Jetbot/jetbot.usd"
                if os.path.exists(jetbot_path):
                    print(f"[Robot] Falling back to Jetbot: {jetbot_path}")
                    self.usd_path = jetbot_path
        
        # USD를 Stage에 추가
        robot_prim_path = "/World/turtlebot3"
        add_reference_to_stage(self.usd_path, robot_prim_path)
        
        # 위치 설정
        robot_prim = XFormPrim(
            prim_path=robot_prim_path,
            name="turtlebot3",
            position=position,
            orientation=orientation,
            scale=(1.0, 1.0, 1.0),
        )
        self.robot = robot_prim
        
        print(f"[Robot] Loaded at position {position}")
        return robot_prim
    
    def setup_lidar(self):
        """LiDAR 센서 설정 및 ROS2 브릿지"""
        print("[Sensor] Configuring LiDAR...")
        
        robot_prim_path = "/World/turtlebot3"
        
        self.lidar = LidarRtxSensor(
            prim_path=f"{robot_prim_path}/base_scan/Lidar",
            name="turtlebot_lidar",
            translation=(-0.032, 0.0, 0.077),
            rotation=(0.0, 0.0, 0.0),
            min_range=0.15,
            max_range=3.5,
            horizontal_fov=360.0,
            horizontal_resolution=1.0,  # degree
            rotation_rate=5.0,  # Hz (LDS-01 사양)
            enable_visualization=True,
        )
        
        # ROS2 Publisher 설정
        self.lidar.set_up_ros2_publisher(topic_name="/scan", frame_id="base_scan")
        
        print("[Sensor] LiDAR → /scan (ROS2 publisher ready)")
        return self.lidar
    
    def setup_ros2_bridge(self):
        """ROS2 Bridge 설정 (OmniGraph)"""
        print("[ROS2] Setting up ROS2 bridge...")
        
        # 체크: ROS2 extension 활성화 확인
        from omni.isaac.ros2_bridge import is_ros2_bridge_available
        if not is_ros2_bridge_available():
            print("[WARN] ROS2 bridge not available. Install ros2 humble first.")
            return False
        
        # OmniGraph으로 ROS2 Subscribe Twist 노드 구성
        # 이 부분은 Isaac Sim 내에서 UI로 설정하거나
        # 아래와 같이 OmniGraph Python API로 설정
        
        og_controller = og.Controller()
        
        # 그래프 생성
        graph_path = "/World/ROS2_TurtleBot_Controller"
        keys = og.Controller.Keys
        (graph, _, _, _) = og_controller.edit(
            graph_path,
            keys.TEMPLATE_NAME,
            "ros2_subscribe_twist",
            og.Graph.template([og.Controller.attribute("inputs:host")])
        )
        
        # ROS2 Subscribe Twist 노드
        (twist_node, _, _) = og_controller.edit(
            f"{graph_path}/SubscribeTwist",
            keys.NODE_NAME,
            "omni.isaac.ros2_bridge.ROS2SubscribeTwist",
            og.Node.template()
        )
        
        # Differential Controller 노드
        (diff_drive_node, _, _) = og_controller.edit(
            f"{graph_path}/DifferentialController",
            keys.NODE_NAME,
            "omni.isaac.core_nodes.DifferentialController",
            og.Node.template({
                "inputs:wheelRadius": 0.033,
                "inputs:wheelDistance": 0.160,
                "inputs:maxLinearVelocity": 0.22,
                "inputs:maxAngularVelocity": 2.84,
            })
        )
        
        # Articulation Controller 노드
        (articulation_node, _, _) = og_controller.edit(
            f"{graph_path}/ArticulationController",
            keys.NODE_NAME,
            "omni.isaac.core_nodes.ArticulationController",
            og.Node.template()
        )
        
        # 노드 연결
        # SubscribeTwist.linearX → DifferentialController.linearVelocity
        # SubscribeTwist.angularZ → DifferentialController.angularVelocity
        # DifferentialController.wheelSpeeds → ArticulationController.velocityCommands
        
        og_controller.connect(
            f"{graph_path}/SubscribeTwist.outputs:linearVelocityX",
            f"{graph_path}/DifferentialController.inputs:linearVelocity"
        )
        og_controller.connect(
            f"{graph_path}/SubscribeTwist.outputs:angularVelocityZ",
            f"{graph_path}/DifferentialController.inputs:angularVelocity"
        )
        
        print("[ROS2] Bridge configured: /cmd_vel → TurtleBot3")
        return True
    
    def setup_obstacles(self):
        """테스트용 장애물 배치"""
        print("[World] Adding test obstacles...")
        
        obstacles = [
            {"pos": (1.5, 0.0, 0.0), "size": (0.3, 0.3, 0.5), "color": (0.8, 0.2, 0.2)},
            {"pos": (-1.0, 1.2, 0.0), "size": (0.4, 0.4, 0.6), "color": (0.2, 0.8, 0.2)},
            {"pos": (0.5, -1.5, 0.0), "size": (0.2, 0.2, 0.4), "color": (0.2, 0.2, 0.8)},
            {"pos": (-1.5, -0.5, 0.0), "size": (0.3, 0.6, 0.5), "color": (0.8, 0.8, 0.2)},
            {"pos": (2.0, 1.0, 0.0), "size": (0.5, 0.2, 0.5), "color": (0.8, 0.2, 0.8)},
        ]
        
        stage = get_current_stage()
        for i, obs in enumerate(obstacles):
            cuboid = DynamicCuboid(
                prim_path=f"/World/obstacle_{i}",
                name=f"obstacle_{i}",
                position=obs["pos"],
                scale=obs["size"],
                color=obs["color"],
                mass=1.0,
            )
            self.world.scene.add(cuboid)
        
        print(f"[World] Added {len(obstacles)} obstacles")
        return True
    
    def setup_navigation_goals(self):
        """Nav Goal 마커 설정"""
        print("[World] Setting navigation goals...")
        
        goals = [
            (3.0, 2.0, 0.0),
            (-2.0, 2.5, 0.0),
            (-2.5, -2.0, 0.0),
            (2.0, -2.5, 0.0),
        ]
        
        stage = get_current_stage()
        for i, goal_pos in enumerate(goals):
            goal_prim = UsdGeom.Sphere.Define(stage, f"/World/goal_{i}")
            goal_prim.AddTranslateOp().Set(Gf.Vec3f(*goal_pos))
            goal_prim.GetRadiusAttr().Set(0.1)
            
            from pxr import UsdLux
            goal_mat = Usd.Prim.Define(stage, f"/World/goal_{i}/material")
            # 녹색 반투명
            goal_mat_attrib = goal_prim.CreateDisplayColorAttr([Gf.Vec3f(0, 1, 0)])
            goal_prim.MakeVisible()
        
        return goals
    
    def run(self, headless: bool = False, num_steps: int = 1000):
        """시뮬레이션 실행"""
        print("[Simulation] Starting...")
        
        self.world.reset()
        
        for step in range(num_steps):
            self.world.step(render=not headless)
            
            if step % 100 == 0:
                robot_pos, _ = self.robot.get_world_pose()
                print(f"  Step {step}: Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
        
        print("[Simulation] Complete!")
    
    def cleanup(self):
        """정리"""
        if self.world:
            self.world.clear()
        print("[Cleanup] Done.")


def main():
    """메인 실행"""
    print("=" * 60)
    print("TurtleBot3 Isaac Sim Simulation Setup")
    print("=" * 60)
    
    # 설정
    usd_path = "/workspace/src/urdf/turtlebot3_burger.usd"
    headless = "--headless" in sys.argv
    
    # 시뮬레이션 인스턴스 생성
    sim = TurtleBotSimulation(usd_path=usd_path, headless=headless)
    
    # 환경 설정
    sim.setup_world()
    sim.load_robot(position=(0.0, 0.0, 0.0))
    sim.setup_lidar()
    sim.setup_ros2_bridge()
    sim.setup_obstacles()
    goals = sim.setup_navigation_goals()
    
    # 시뮬레이션 실행
    sim.run(headless=headless, num_steps=500)
    sim.cleanup()
    
    print("\n✅ Simulation setup complete!")


if __name__ == "__main__":
    main()
```

## 4.3 ROS2 Bridge 상세

### 4.3.1 ROS2 토픽 맵핑

| Isaac Sim ROS2 Node | ROS2 Topic | Type | 방향 | 설명 |
|---------------------|------------|------|------|------|
| ROS2SubscribeTwist | `/cmd_vel` | `Twist` | Input | 속도 명령 수신 |
| ROS2PublishJointState | `/joint_states` | `JointState` | Output | 조인트 상태 |
| ROS2PublishOdometry | `/odom` | `Odometry` | Output | 오도메트리 |
| ROS2PublishTF | `/tf` | `TFMessage` | Output | 좌표 변환 |
| ROS2PublishLaserScan | `/scan` | `LaserScan` | Output | LiDAR 스캔 |
| ROS2PublishCamera | `/camera/image_raw` | `Image` | Output | 카메라 이미지 |
| ROS2PublishIMU | `/imu` | `Imu` | Output | IMU 데이터 |

### 4.3.2 수동 주행 테스트

```bash
# Terminal 1: Isaac Sim 실행 (Docker)
./isaac_sim_docker.sh
# Isaac Sim 내에서 TurtleBot USD 로드 및 ROS2 Bridge 활성화

# Terminal 2: ROS2 키보드 텔레옵
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 3: 토픽 모니터링
source /opt/ros/humble/setup.bash
ros2 topic echo /odom
```

### 4.3.3 Isaac Sim → ROS2 데이터 흐름

```
Isaac Sim (Omniverse)
│
├── PhysX Physics → ArticulationController → Joint States
│                                              │
├── LiDAR Sensor → LaserScan → ROS2PublishLaserScan → /scan
│                                                      │
├── Odometry → ROS2PublishOdometry → /odom
│                                      │
├── TF → ROS2PublishTF → /tf
│                          │
└── Camera → ROS2PublishCamera → /camera/image_raw
                                   │
                                   ▼
                        ROS2 Network (DDS)
                                   │
                                   ▼
                        Nav2 / RViz / RQT
```

## 4.4 Nav2 통합

### 4.4.1 Nav2 런치

```python
"""src/deployment/nav2_turtlebot_launch.py"""
# ROS2 launch 파일로 Isaac Sim + Nav2 통합
# 실제 ROS2 노드로 실행

import launch
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Nav2 nodes
    nav2_nodes = [
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'autostart': True,
                        'node_names': ['controller_server', 'planner_server',
                                       'behavior_server', 'bt_navigator']}],
        ),
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ]
    
    return launch.LaunchDescription(nav2_nodes)
```

### 4.4.2 Nav2 파라미터

**`config/nav2_params.yaml`**:
```yaml
controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    controller_plugins: ["FollowPath"]
    
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.18
      max_linear_accel: 0.15
      max_linear_decel: 0.20
      lookahead_dist: 0.4
      min_lookahead_dist: 0.3
      max_lookahead_dist: 0.5
      rotate_to_heading_angular_vel: 1.0
      max_angular_accel: 0.5
      use_velocity_scaled_lookahead: true
      use_regulated_linear_velocity_scaling: true
      use_cost_regulated_linear_velocity_scaling: true
      regulated_linear_scaling_min_radius: 0.15
      regulated_linear_scaling_min_speed: 0.05
      use_rotate_to_heading: true
      allow_reversing: false
      rotate_to_heading_min_angle: 0.5
      max_robot_pose_search_dist: 10.0

planner_server:
  ros__parameters:
    use_sim_time: True
    planner_plugins: ["GridBased"]
    
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.15
      use_astar: false
      allow_unknown: true

bt_navigator:
  ros__parameters:
    use_sim_time: True
    bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialized_pose_condition_bt_node
      - nav2_distance_traveled_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_areapass_condition_bt_node
      - nav2_globally_updated_goal_condition_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_bt_node
      - nav2_goal_updated_bt_node
```

## 4.5 Isaac Sim 주요 단축키

| 동작 | 단축키 |
|------|--------|
| Viewport 회전 | LMB 드래그 |
| Viewport 이동 | MMB 드래그 |
| Viewport 확대/축소 | 스크롤 |
| 게임 뷰 (시점 잠금) | RMB 드래그 |
| 객체 선택 | Ctrl + LMB |
| Transform 조작 | R (회전), T (이동), Y (스케일) |
| Material 변경 | Ctrl + M |
| USD Stage 열기 | Ctrl + S |
| Play/Pause | Space |
| Step Frame | Ctrl + → |

## 4.6 문제 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| LiDAR scan 안 옴 | ROS2 Bridge 미활성화 | `Window → ROS2 → Enable ROS2 Bridge` |
| /cmd_vel 응답 없음 | Articulation Controller 미연결 | OmniGraph에서 Controller 연결 확인 |
| 로봇 무게 안 잡힘 | Fixed Base 체크됨 | URDF Import 시 체크 해제 |
| 바퀴만 움직이고 로봇 미이동 | 마찰 계수 문제 | PhysX 마찰 설정 확인 |
| 성능 저하 | GPU 메모리 부족 | Viewport 품질 낮춤 (Settings → Render → Quality) |
| ROS2 통신 안 됨 | Domain ID 불일치 | `ROS_DOMAIN_ID` 환경변수 통일 |
