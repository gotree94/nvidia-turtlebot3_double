# 03. URDF 로봇 모델링

> **TurtleBot3 URDF 구조 분석 및 Isaac Sim 변환**

## 3.1 TurtleBot3 Burger URDF 구조

TurtleBot3 Burger의 URDF는 다음 구조로 구성됩니다:

```
turtlebot3_burger.urdf
├── base_link                # 로봇 베이스 (본체)
├── wheel_left_link          # 왼쪽 바퀴 (continuous joint)
├── wheel_right_link         # 오른쪽 바퀴 (continuous joint)
├── caster_back_link         # 뒷면 캐스터 (fixed joint)
├── caster_front_link        # 앞면 캐스터 (fixed joint) [Waffle only]
├── imu_link                 # IMU 센서 (fixed joint)
├── base_scan                # LiDAR 센서 (fixed joint)
├── camera_link              # 카메라 (fixed joint, 선택)
├── camera_depth_frame       # Depth 카메라 (선택)
└── plate_link               # 상판 (fixed joint)
```

### 3.1.1 TurtleBot3 Burger 주요 치수

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 휠베이스 (wheel separation) | 0.160 m | 좌우 바퀴 간 거리 |
| 휠 반경 | 0.033 m | 바퀴 반지름 |
| 로봇 폭 | 0.140 m | base_link collision box width |
| 로봇 높이 | 0.143 m | base_link collision box height |
| LiDAR 높이 | ~0.080 m | 바닥으로부터 LiDAR 높이 |
| 최대 속도 | 0.22 m/s | 권장 최대 선속도 |
| 최대 각속도 | 2.84 rad/s | 권장 최대 각속도 |

## 3.2 URDF 파일: TurtleBot3 Burger

**`src/urdf/turtlebot3_burger.urdf`**:

```xml
<?xml version="1.0" encoding="utf-8"?>
<!-- ============================================================
     TurtleBot3 Burger URDF for NVIDIA Isaac Sim
     Reference: ROBOTIS-GIT/turtlebot3/turtlebot3_description/urdf
     Modified for Isaac Sim compatibility
     ============================================================ -->
<robot name="turtlebot3_burger" xmlns:xacro="http://ros.org/wiki/xacro">

  <!-- ==================== Material Definitions ==================== -->
  <material name="light_black">
    <color rgba="0.3 0.3 0.3 1.0"/>
  </material>
  <material name="dark">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>

  <!-- ==================== Base Link (Robot Body) ==================== -->
  <link name="base_link">
    <visual>
      <origin rpy="0 0 0" xyz="-0.032 0 0.0"/>
      <geometry>
        <mesh filename="package://turtlebot3_description/meshes/bases/burger_base.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="light_black"/>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="-0.032 0 0.070"/>
      <geometry>
        <box size="0.140 0.140 0.143"/>
      </geometry>
    </collision>
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.8257"/>
      <inertia
        ixx="0.002212" ixy="-0.000012" ixz="0.000035"
        iyy="0.002119" iyz="-0.000005"
        izz="0.002006"/>
    </inertial>
  </link>

  <!-- ==================== Left Wheel ==================== -->
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left_link"/>
    <origin rpy="-1.57 0 0" xyz="0.0 0.080 0.023"/>
    <axis xyz="0 0 1"/>
    <limit effort="1.0" velocity="10.0"/>
    <dynamics damping="0.001" friction="0.001"/>
  </joint>
  <link name="wheel_left_link">
    <visual>
      <origin rpy="1.57 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="package://turtlebot3_description/meshes/wheels/left_tire.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="dark"/>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <cylinder length="0.018" radius="0.033"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.0285"/>
      <inertia
        ixx="0.000011" ixy="0.0" ixz="0.0"
        iyy="0.000011" iyz="0.0"
        izz="0.000021"/>
    </inertial>
  </link>

  <!-- ==================== Right Wheel ==================== -->
  <joint name="wheel_right_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right_link"/>
    <origin rpy="-1.57 0 0" xyz="0.0 -0.080 0.023"/>
    <axis xyz="0 0 1"/>
    <limit effort="1.0" velocity="10.0"/>
    <dynamics damping="0.001" friction="0.001"/>
  </joint>
  <link name="wheel_right_link">
    <visual>
      <origin rpy="1.57 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="package://turtlebot3_description/meshes/wheels/right_tire.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="dark"/>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <cylinder length="0.018" radius="0.033"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.0285"/>
      <inertia
        ixx="0.000011" ixy="0.0" ixz="0.0"
        iyy="0.000011" iyz="0.0"
        izz="0.000021"/>
    </inertial>
  </link>

  <!-- ==================== Caster Back ==================== -->
  <joint name="caster_back_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_back_link"/>
    <origin rpy="-1.57 0 0" xyz="-0.081 0 -0.004"/>
  </joint>
  <link name="caster_back_link">
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.014"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.001"/>
      <inertia
        ixx="0.000001" ixy="0.0" ixz="0.0"
        iyy="0.000001" iyz="0.0"
        izz="0.000001"/>
    </inertial>
  </link>

  <!-- ==================== LiDAR (LDS-01) ==================== -->
  <joint name="base_scan_joint" type="fixed">
    <parent link="base_link"/>
    <child link="base_scan"/>
    <origin rpy="0 0 0" xyz="-0.032 0 0.077"/>
  </joint>
  <link name="base_scan">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="package://turtlebot3_description/meshes/sensors/lds.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="dark"/>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <cylinder length="0.030" radius="0.036"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.010"/>
      <inertia
        ixx="0.000001" ixy="0.0" ixz="0.0"
        iyy="0.000001" iyz="0.0"
        izz="0.000001"/>
    </inertial>
  </link>

  <!-- ==================== IMU Link ==================== -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
  </joint>
  <link name="imu_link">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.005"/>
      <inertia
        ixx="0.000001" ixy="0.0" ixz="0.0"
        iyy="0.000001" iyz="0.0"
        izz="0.000001"/>
    </inertial>
  </link>

  <!-- ==================== Plate (상판) ==================== -->
  <joint name="plate_joint" type="fixed">
    <parent link="base_link"/>
    <child link="plate_link"/>
    <origin rpy="0 0 0" xyz="-0.032 0 0.070"/>
  </joint>
  <link name="plate_link">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="package://turtlebot3_description/meshes/bases/burger_plate.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="white"/>
    </visual>
  </link>

  <!-- ==================== Camera (선택: USB Camera 모델링) ==================== -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin rpy="0 0 0" xyz="0.040 0 0.060"/>
  </joint>
  <link name="camera_link">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.025 0.020 0.020"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.025 0.020 0.020"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.005"/>
      <inertia
        ixx="0.000001" ixy="0.0" ixz="0.0"
        iyy="0.000001" iyz="0.0"
        izz="0.000001"/>
    </inertial>
  </link>

  <!-- ==================== ROS2 Gazebo Plugins ==================== -->
  <gazebo>
    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find turtlebot3_description)/config/turtlebot3_ros2_control.yaml</parameters>
    </plugin>
  </gazebo>

</robot>
```

## 3.3 URDF → USD 변환 (Isaac Sim)

Isaac Sim은 URDF 파일을 USD(Universal Scene Description)로 변환하여 사용합니다.

### 3.3.1 URDF Importer 사용 방법

**Isaac Sim UI를 통한 변환:**

1. Isaac Sim 실행
2. Menu: `Isaac Utils → URDF Importer`
3. URDF 파일 선택: `turtlebot3_burger.urdf`
4. Import 설정:
   - **☐ Fix Base Link** → **체크 해제** (모바일 로봇은 바퀴가 움직여야 함)
   - **Joint Drive Type** → `Velocity` (속도 제어)
   - **Make default prim** → 체크
   - **Stage in meters** → 체크
5. `Import` 버튼 클릭
6. USD 파일 저장: `File → Save As → turtlebot3_burger.usd`

### 3.3.2 Python 스크립트를 통한 변환

**`src/isaac_sim/urdf_to_usd.py`**:

```python
"""
TurtleBot3 URDF를 Isaac Sim USD로 변환하는 스크립트

Usage:
    # Isaac Sim 내에서 실행:
    ./python.sh src/isaac_sim/urdf_to_usd.py
    
    # 또는 컨테이너 내에서:
    /isaac-sim/python.sh /workspace/src/isaac_sim/urdf_to_usd.py
"""

import os
import sys
import carb
import omni.usd
from pxr import Usd, UsdGeom, Sdf, Gf


def import_urdf_to_usd(urdf_path: str, output_usd_path: str, fix_base: bool = False):
    """
    URDF 파일을 USD 파일로 변환합니다.
    
    Args:
        urdf_path: 입력 URDF 파일 경로
        output_usd_path: 출력 USD 파일 경로
        fix_base: 베이스 링크 고정 여부 (mobile robot = False)
    """
    from omni.isaac.urdf import _urdf
    
    urdf_importer = _urdf.acquire_urdf_interface()
    
    # Import 설정
    import_config = omni.isaac.urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decompose_mesh = False
    import_config.fix_base = fix_base
    import_config.make_default_prim = True
    import_config.distance_scale = 1.0  # meter 단위
    import_config.joint_drive_type = "velocity"  # 속도 제어 모드
    import_config.drive_dofs = True
    
    print(f"[URDF Import] Loading URDF: {urdf_path}")
    result = urdf_importer.import_urdf(urdf_path, import_config)
    
    if not result.valid:
        print(f"[ERROR] URDF import failed: {result.error_message}")
        return False
    
    stage = omni.usd.get_context().get_stage()
    
    # USD 파일 저장
    usd_file_path = output_usd_path
    stage.GetRootLayer().Export(usd_file_path)
    print(f"[USD Export] Saved to: {usd_file_path}")
    
    return True


def add_turtlebot_wheel_controllers(stage, robot_prim_path: str = "/turtlebot3_burger"):
    """
    TurtleBot3의 Differential Controller 설정을 USD에 추가합니다.
    """
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics, Gf
    
    # Differential Controller 구성
    wheel_radius = 0.033
    wheel_separation = 0.160
    
    print(f"[Controller] Configuring Differential Drive:")
    print(f"  Wheel Radius: {wheel_radius}m")
    print(f"  Wheel Separation: {wheel_separation}m")
    print(f"  Robot Prim: {robot_prim_path}")
    
    # 여기서 Articulation Controller + Differential Controller
    # OmniGraph 노드를 구성합니다 (Isaac Sim UI 또는 Python OmniGraph API 사용)
    
    return True


if __name__ == "__main__":
    # URDF 경로 (실제 환경에 맞게 조정)
    urdf_path = "/workspace/src/urdf/turtlebot3_burger.urdf"
    output_path = "/workspace/src/urdf/turtlebot3_burger.usd"
    
    # 변환 실행
    success = import_urdf_to_usd(
        urdf_path=urdf_path,
        output_usd_path=output_path,
        fix_base=False  # 모바일 로봇 = False
    )
    
    if success:
        print("\n✅ URDF to USD conversion successful!")
        print(f"   Output: {output_path}")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)
```

### 3.3.3 xacro → URDF 변환

ROBOTIS-GIT 저장소의 xacro 파일을 URDF로 변환:

```bash
# ROS2 패키지에서 직접 변환
source /opt/ros/humble/setup.bash

# TurtleBot3 Burger
cd /workspace/src/urdf
xacro $(ros2 pkg prefix turtlebot3_description)/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro \
    > turtlebot3_burger.urdf

# TurtleBot3 Waffle
xacro $(ros2 pkg prefix turtlebot3_description)/share/turtlebot3_description/urdf/turtlebot3_waffle.urdf.xacro \
    > turtlebot3_waffle.urdf

# 처리된 파일에서 ROS 패키지 경로 제거
sed -i 's|package://turtlebot3_description/||g' turtlebot3_burger.urdf
```

## 3.4 Isaac Sim TurtleBot3 컨트롤러 구성

### 3.4.1 Differential Drive Controller

Isaac Sim에서 TurtleBot3의 차동 구동을 위해 다음 OmniGraph 노드가 필요합니다:

| 노드 | 역할 | 설정값 |
|------|------|--------|
| **ROS2 Subscribe Twist** | `/cmd_vel` 구독 | linear/x, angular/z |
| **Differential Controller** | 차동 속도 → 휠 속도 | wheel_radius=0.033, wheel_separation=0.160 |
| **Articulation Controller** | 휠 조인트 명령 | velocity commands |

### 3.4.2 LiDAR 센서 구성

```python
"""
Isaac Sim에서 TurtleBot3 LiDAR 설정
"""
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import LidarRtxSensor


def setup_turtlebot_lidar(stage, robot_prim_path: str):
    """TurtleBot3에 LiDAR 센서를 부착합니다."""
    
    lidar_config = {
        "name": "turtlebot_lidar",
        "prim_path": f"{robot_prim_path}/base_scan/Lidar",
        "translation": (-0.032, 0.0, 0.077),
        "rotation": (0.0, 0.0, 0.0),
        "min_range": 0.15,
        "max_range": 3.5,
        "horizontal_resolution": 1.0,  # degree
        "horizontal_fov": 360.0,
        "rotation_speed": 5.0,  # Hz
        "enable_visualization": True,
    }
    
    lidar = LidarRtxSensor(
        prim_path=lidar_config["prim_path"],
        name=lidar_config["name"],
        translation=lidar_config["translation"],
        rotation=lidar_config["rotation"],
        min_range=lidar_config["min_range"],
        max_range=lidar_config["max_range"],
        horizontal_fov=lidar_config["horizontal_fov"],
        horizontal_resolution=lidar_config["horizontal_resolution"],
        rotation_rate=lidar_config["rotation_speed"],
    )
    
    # LiDAR 데이터를 ROS2 /scan 토픽으로 publish
    lidar.set_up_ros2_publisher(topic_name="/scan", frame_id="base_scan")
    
    print(f"[LiDAR] Configured TurtleBot3 LiDAR:")
    print(f"  Range: {lidar_config['min_range']} - {lidar_config['max_range']}m")
    print(f"  FOV: {lidar_config['horizontal_fov']}°")
    print(f"  Resolution: {lidar_config['horizontal_resolution']}°")
    
    return lidar
```

## 3.5 Isaac Sim용 TurtleBot3 USD 참고

변환 완료 후 USD 파일은 다음과 같은 계층 구조를 가집니다:

```
/turtlebot3_burger (Xform)
├── base_link (Xform) [RigidBody]
│   ├── base_link_visual (Mesh)
│   └── base_link_collision (Mesh)
├── wheel_left_link (Xform) [RigidBody]
│   ├── wheel_left_link_visual (Mesh)
│   └── wheel_left_link_collision (Mesh)
├── wheel_right_link (Xform) [RigidBody]
│   ├── wheel_right_link_visual (Mesh)
│   └── wheel_right_link_collision (Mesh)
├── base_scan (Xform) [RigidBody]
│   ├── base_scan_visual (Mesh)
│   └── base_scan_collision (Mesh)
├── caster_back_link (Xform)
├── imu_link (Xform)
├── plate_link (Xform)
├── camera_link (Xform)
├── wheel_left_joint (PhysicsRevoluteJoint)
├── wheel_right_joint (PhysicsRevoluteJoint)
└── ...
```

## 3.6 미터 체계 일관성

Isaac Sim은 **미터(m)** 단위를 사용합니다. TurtleBot3의 URDF 기본 단위도 미터이므로 추가 변환이 필요 없습니다.

| 변환 대상 | URDF | Isaac Sim |
|-----------|------|-----------|
| 휠 반경 | 0.033 m | 0.033 |
| 휠베이스 | 0.160 m | 0.160 |
| LiDAR 범위 | 0.15 ~ 3.5 m | 0.15 ~ 3.5 |
| 질량 | kg | kg |
| 관성 모멘트 | kg·m² | kg·m² |

## 3.7 Isaac Lab에서 TurtleBot3 환경 설정

Isaac Lab은 직접적인 URDF 지원 대신, 환경 설정에서 로봇의 속성을 정의합니다.

**`src/isaac_lab/turtlebot3_cfg.py`**:

```python
"""
Isaac Lab용 TurtleBot3 Configuration
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TurtleBot3Config:
    """TurtleBot3 Burger 기본 설정"""
    
    # 로봇 물리
    robot_name: str = "turtlebot3_burger"
    usd_path: str = "/workspace/src/urdf/turtlebot3_burger.usd"
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # 차동 구동
    wheel_radius: float = 0.033      # m
    wheel_separation: float = 0.160  # m
    max_linear_velocity: float = 0.22   # m/s
    max_angular_velocity: float = 2.84  # rad/s
    
    # LiDAR
    lidar_min_range: float = 0.15    # m
    lidar_max_range: float = 3.5     # m
    lidar_num_rays: int = 360        # 1° resolution
    lidar_horizontal_fov: float = 360.0  # degree
    
    # Motion limits
    linear_velocity_scale: float = 1.0
    angular_velocity_scale: float = 1.0
    
    # Collision geometry
    base_radius: float = 0.070       # m (half of 0.140)
    base_height: float = 0.143       # m
```

## 3.8 URDF 커스터마이징

### 3.8.1 Jetson Orin Nano 추가 (Robot 1)

Jetson Orin Nano를 TurtleBot3 상판에 장착하는 경우 URDF에 추가:

```xml
<!-- Jetson Orin Nano 추가 -->
<joint name="jetson_joint" type="fixed">
    <parent link="plate_link"/>
    <child link="jetson_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.0 0.020"/>
</joint>
<link name="jetson_link">
    <visual>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <box size="0.100 0.080 0.020"/>
        </geometry>
        <material name="red"/>
    </visual>
    <collision>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <box size="0.100 0.080 0.020"/>
        </geometry>
    </collision>
    <inertial>
        <origin xyz="0 0 0"/>
        <mass value="0.250"/>  <!-- Jetson Orin Nano ~250g -->
        <inertia
            ixx="0.0003" ixy="0.0" ixz="0.0"
            iyy="0.0003" iyz="0.0"
            izz="0.0001"/>
    </inertial>
</link>
```

### 3.8.2 Raspberry Pi 5 추가 (Robot 2)

```xml
<!-- Raspberry Pi 5 추가 -->
<joint name="rpi_joint" type="fixed">
    <parent link="plate_link"/>
    <child link="rpi_link"/>
    <origin rpy="0 0 0" xyz="0.0 0.0 0.015"/>
</joint>
<link name="rpi_link">
    <visual>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <box size="0.085 0.056 0.015"/>
        </geometry>
        <material name="red"/>
    </visual>
    <collision>
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry>
            <box size="0.085 0.056 0.015"/>
        </geometry>
    </collision>
    <inertial>
        <origin xyz="0 0 0"/>
        <mass value="0.045"/>  <!-- RPi 5 ~45g -->
        <inertia
            ixx="0.00002" ixy="0.0" ixz="0.0"
            iyy="0.00002" iyz="0.0"
            izz="0.00001"/>
    </inertial>
</link>
```

## 3.9 참고: Gazebo → Isaac Sim 차이점

| 항목 | Gazebo | Isaac Sim |
|------|--------|-----------|
| 물리 엔진 | DART/ODE/Bullet | PhysX 5 |
| 링크 고정 | `<gazebo>` 태그 내부 | USD Physics 속성 |
| 플러그인 | `<gazebo>` plugin | OmniGraph 노드 |
| ROS2 통신 | `gazebo_ros2_*` 플러그인 | 내장 ROS2 Bridge |
| 조인트 드라이브 | `<transmission>` 태그 | USD Physics Drive API |
| URDF 변환 | xacro → URDF → SDF | URDF → USD 직접 변환 |
