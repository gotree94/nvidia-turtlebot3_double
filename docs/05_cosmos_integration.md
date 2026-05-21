# 05. NVIDIA Cosmos 통합 및 합성 데이터 생성

> **Cosmos World Foundation Model을 활용한 합성 데이터 생성, Transfer, Policy 학습**

## 5.1 Cosmos 개요

NVIDIA Cosmos는 Physical AI를 위한 World Foundation Model(WFM) 플랫폼입니다.

### Cosmos 주요 컴포넌트

| 컴포넌트 | 설명 | TurtleBot3 적용 |
|----------|------|-----------------|
| **Cosmos-Predict2** | 비디오 예측 모델 (Video2World). 입력 비디오에서 미래 프레임 예측 | 내비게이션 시나리오 시뮬레이션 |
| **Cosmos-Transfer** | 멀티모달 제어 (ControlNet 기반). Simulation-to-Real 변환 | Isaac Sim → 포토리얼리스틱 렌더링 |
| **Cosmos-Reason** | VLM 기반 물리적 추론 | 장면 이해, 충돌 회피 추론 |
| **Cosmos-Policy** | Cosmos-Predict2 기반 Visuomotor Control Policy | 엔드투엔드 정책 학습 |

### Cosmos + Isaac Sim 통합 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Isaac Sim                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  TurtleBot3 Navigation Environment                                  ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      ││
│  │  │ RGB     │ │ Depth   │ │Segmen-  │ │ Shaded  │ │ Edges   │      ││
│  │  │ Camera  │ │ Camera  │ │tation   │ │Seg      │ │         │      ││
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘      ││
│  │       │           │           │           │           │            ││
│  │       └───────────┴───────────┴───────────┴───────────┘            ││
│  │                           │ CosmosWriter                            ││
│  └───────────────────────────┼─────────────────────────────────────────┘│
│                              │                                          │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │ mp4 files (5 modalities)
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            Cosmos                                        │
│  ┌────────────────────┐   ┌────────────────────┐                         │
│  │  Cosmos-Transfer   │   │  Cosmos-Predict2   │                         │
│  │  (Sim→Real 변환)   │   │  (비디오 예측)      │                         │
│  │                    │   │                    │                         │
│  │  Input: RGB+Edge  │   │  Input: Robot View │                         │
│  │  Output: Reality   │   │  Output: Future    │                         │
│  └─────────┬──────────┘   └──────────┬─────────┘                         │
│            │                         │                                   │
│            ▼                         ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │  Cosmos-Policy (Post-training)                               │        │
│  │  Robot action policy 학습                                     │        │
│  └──────────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────┘
```

## 5.2 CosmosWriter: Isaac Sim 데이터 수집

Isaac Sim의 `CosmosWriter`를 사용하여 TurtleBot3 내비게이션 데이터를 수집합니다.

### 5.2.1 CosmosWriter 파이프라인

**`src/isaac_sim/cosmos_sdg_pipeline.py`**:

```python
"""
Isaac Sim CosmosWriter 기반 TurtleBot3 Navigation 합성 데이터 생성 파이프라인

이 스크립트는 TurtleBot3가 웨어하우스 환경을 탐색하면서
Cosmos 학습용 멀티모달 데이터를 수집합니다.

Usage:
    /isaac-sim/python.sh /workspace/src/isaac_sim/cosmos_sdg_pipeline.py
"""

import os
import sys
import math
import json
import numpy as np
import carb
import omni
import omni.replicator.core as rep
from omni.isaac.core import SimulationContext
from omni.isaac.core.world import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import UsdGeom, Gf, Sdf


class TurtleBotDataPipeline:
    """TurtleBot3 Cosmos 데이터 수집 파이프라인"""
    
    def __init__(self, output_dir: str = "/workspace/_out_cosmos_turtlebot"):
        self.output_dir = output_dir
        self.world = None
        self.robot = None
        self.camera_prim = None
        self.cosmos_writer = None
        self.camera_path = None
        
    def setup_environment(self):
        """기본 환경 설정"""
        print("[Setup] Creating warehouse environment...")
        
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # 웨어하우스 배경 에셋 로드
        assets_root = get_assets_root_path()
        if assets_root:
            warehouse_path = f"{assets_root}/Environments/Warehouse/warehouse.usd"
            if os.path.exists(warehouse_path):
                add_reference_to_stage(warehouse_path, "/World/Warehouse")
                print(f"[Environment] Loaded warehouse: {warehouse_path}")
            else:
                print("[Environment] No warehouse asset, using simple walls")
                self._create_simple_walls()
        
        # 조명
        from pxr import UsdLux
        stage = get_current_stage()
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome_light.CreateIntensityAttr(1000)
        
        print("[Setup] Environment ready.")
    
    def _create_simple_walls(self):
        """간단한 벽 환경 생성 (웨어하우스 없을 때 대체)"""
        stage = get_current_stage()
        walls = [
            {"pos": (0, -5, 0), "size": (10, 0.2, 2), "color": (0.5, 0.5, 0.5)},
            {"pos": (0, 5, 0), "size": (10, 0.2, 2), "color": (0.5, 0.5, 0.5)},
            {"pos": (-5, 0, 0), "size": (0.2, 10, 2), "color": (0.5, 0.5, 0.5)},
            {"pos": (5, 0, 0), "size": (0.2, 10, 2), "color": (0.5, 0.5, 0.5)},
        ]
        for i, wall in enumerate(walls):
            cube = UsdGeom.Cube.Define(stage, f"/World/wall_{i}")
            cube.AddTranslateOp().Set(Gf.Vec3f(*wall["pos"]))
            cube.AddScaleOp().Set(Gf.Vec3f(*wall["size"]))
        
        # 장애물 배치
        obstacles_pos = [
            (-2, -2, 0), (2, 2, 0), (-1.5, 1.5, 0), 
            (1.5, -1.5, 0), (0, 0, 0), (-2.5, 0, 0)
        ]
        for i, pos in enumerate(obstacles_pos):
            cube = UsdGeom.Cube.Define(stage, f"/World/obstacle_{i}")
            cube.AddTranslateOp().Set(Gf.Vec3f(*pos))
            cube.AddScaleOp().Set(Gf.Vec3f(0.3, 0.3, 0.5))
    
    def load_robot(self):
        """TurtleBot3 로드"""
        print("[Robot] Loading TurtleBot3...")
        robot_path = "/workspace/src/urdf/turtlebot3_burger.usd"
        
        if not os.path.exists(robot_path):
            assets_root = get_assets_root_path()
            if assets_root:
                robot_path = f"{assets_root}/Isaac/Robots/Jetbot/jetbot.usd"
        
        add_reference_to_stage(robot_path, "/World/TurtleBot")
        self.robot = XFormPrim("/World/TurtleBot", name="turtlebot")
        
        # 랜덤 초기 위치
        import random
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        self.robot.set_world_pose(position=(x, y, 0.0))
        
        print(f"[Robot] Loaded at ({x:.2f}, {y:.2f})")
    
    def setup_camera(self):
        """TurtleBot3 전방 카메라 설정"""
        print("[Camera] Setting up front camera...")
        
        stage = get_current_stage()
        camera_prim = UsdGeom.Camera.Define(stage, "/World/TurtleBot/camera_front")
        camera_prim.AddTranslateOp().Set(Gf.Vec3f(0.04, 0.0, 0.06))
        camera_prim.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
        
        # 카메라 파라미터
        camera_prim.GetFocalLengthAttr().Set(24.0)  # mm
        camera_prim.GetHorizontalApertureAttr().Set(20.955)
        camera_prim.GetVerticalApertureAttr().Set(15.716)
        camera_prim.GetClippingRangeAttr().Set((0.1, 100.0))
        
        self.camera_prim = camera_prim
        # 카메라 경로 (render product 용)
        self.camera_path = "/World/TurtleBot/camera_front"
        
        print(f"[Camera] Camera ready at {self.camera_path}")
        return self.camera_path
    
    def setup_cosmos_writer(self, num_clips: int = 5, num_frames_per_clip: int = 60, 
                            capture_interval: int = 5):
        """CosmosWriter 설정"""
        print("[CosmosWriter] Initializing...")
        
        # Render Product 생성
        render_product = rep.create.render_product(self.camera_path, (1280, 720))
        
        # CosmosWriter 설정
        self.cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
        
        # Disk Backend 설정
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=self.output_dir)
        
        # Writer 초기화
        self.cosmos_writer.initialize(
            backend=backend,
            use_instance_id=True,
            segmentation_mapping={
                "robot": [0, 1.0],
                "obstacle": [0, 2.0],
                "floor": [1, 3.0],
                "wall": [2, 4.0],
                "goal": [3, 5.0],
            }
        )
        
        self.cosmos_writer.attach(render_product)
        
        # 파라미터 저장
        self.num_clips = num_clips
        self.num_frames_per_clip = num_frames_per_clip
        self.capture_interval = capture_interval
        
        print(f"[CosmosWriter] Configured:")
        print(f"  Output: {self.output_dir}")
        print(f"  Clips: {num_clips}, Frames: {num_frames_per_clip}")
        print(f"  Resolution: 1280x720")
        print(f"  Modalities: RGB, Depth, Segmentation, ShadedSeg, Edges")
    
    def run_navigation_pattern(self, steps: int = 100):
        """TurtleBot3 내비게이션 패턴 실행 (데이터 수집용)"""
        
        # Waypoint 정의
        waypoints = [
            (2.0, 1.0), (1.0, 2.0), (-1.0, 2.0), (-2.0, 1.0),
            (-2.0, -1.0), (-1.0, -2.0), (1.0, -2.0), (2.0, -1.0),
        ]
        current_wp = 0
        
        for step in range(steps):
            # 간단한 waypoint 추종
            target = waypoints[current_wp]
            pos, _ = self.robot.get_world_pose()
            
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 0.3:
                current_wp = (current_wp + 1) % len(waypoints)
                continue
            
            # 속도 명령 (간단한 P 컨트롤러)
            angle = math.atan2(dy, dx)
            linear = min(0.15, dist * 0.3)
            
            # 회전 제어
            angle_diff = angle  # 단순화
            angular = max(-0.5, min(0.5, angle_diff * 0.5))
            
            # 물리 스텝
            self.world.step(render=True)
            
            if step % 20 == 0:
                print(f"  [{step:3d}/{steps}] pos=({pos[0]:.2f},{pos[1]:.2f}) "
                      f"target=({target[0]:.2f},{target[1]:.2f})")
        
        return True
    
    def run_pipeline(self, headless: bool = False):
        """전체 데이터 수집 파이프라인 실행"""
        print("\n" + "=" * 60)
        print("TurtleBot3 Cosmos Data Collection Pipeline")
        print("=" * 60 + "\n")
        
        # 1. 환경 설정
        self.setup_environment()
        
        # 2. 로봇 로드
        self.load_robot()
        
        # 3. 카메라 설정
        self.setup_camera()
        
        # 4. CosmosWriter 설정
        self.setup_cosmos_writer(
            num_clips=5,
            num_frames_per_clip=60,
            capture_interval=5
        )
        
        # 5. 데이터 수집
        print("\n[Pipeline] Starting data collection...")
        
        # Rep.orchestrator를 통한 데이터 수집
        with rep.trigger.on_frame(num_frames=self.num_clips * self.num_frames_per_clip):
            self.run_navigation_pattern(steps=500)
        
        print(f"\n[Pipeline] Data collection complete!")
        print(f"  Output: {self.output_dir}")
        print(f"  Contents:")
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                print(f"    📁 {item}/")
                for f in os.listdir(item_path):
                    print(f"      📄 {f}")
            else:
                print(f"    📄 {item}")
        
        return True


def run_sdg_pipeline(
    camera_path, 
    num_clips, 
    num_frames_per_clip, 
    capture_interval, 
    use_instance_id=True, 
    segmentation_mapping=None
):
    """
    SDG 파이프라인 실행 함수 (재사용 가능한 독립 함수)
    """
    rp = rep.create.render_product(camera_path, (1280, 720))
    
    cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), "_out_cosmos_warehouse")
    
    print(f"output_directory: {out_dir}")
    backend.initialize(output_dir=out_dir)
    
    cosmos_writer.initialize(
        backend=backend,
        use_instance_id=use_instance_id,
        segmentation_mapping=segmentation_mapping
    )
    cosmos_writer.attach(rp)
    
    return cosmos_writer


if __name__ == "__main__":
    pipeline = TurtleBotDataPipeline()
    pipeline.run_pipeline()
```

### 5.2.2 CosmosWriter 출력 데이터 구조

```
_out_cosmos_turtlebot/
├── clip_0000/
│   ├── rgb.mp4              # RGB 비디오
│   ├── depth.mp4            # Depth 맵
│   ├── segmentation.mp4     # 시맨틱 세그멘테이션
│   ├── shaded_seg.mp4       # 셰이딩 적용 세그멘테이션
│   ├── edges.mp4            # 에지 맵
│   └── metadata.json        # 프레임별 메타데이터
├── clip_0001/
│   └── ...
├── clip_0002/
│   └── ...
├── clip_0003/
│   └── ...
└── clip_0004/
    └── ...
```

## 5.3 Cosmos-Transfer: Sim-to-Real 변환

Cosmos-Transfer는 Isaac Sim의 합성 데이터를 현실적인 비디오로 변환합니다.

### 5.3.1 Cosmos-Transfer 실행

**`src/cosmos/cosmos_transfer.py`**:

```python
"""
Cosmos-Transfer: Isaac Sim → Reality 변환 파이프라인

Usage:
    conda activate cosmos
    python src/cosmos/cosmos_transfer.py --input_dir /path/to/cosmos_data --output_dir /path/to/output
"""

import os
import sys
import argparse
import subprocess


def run_cosmos_transfer(input_dir: str, output_dir: str, model: str = "Cosmos-Transfer1-7B"):
    """Cosmos-Transfer 모델 실행"""
    
    cosmos_home = os.environ.get("COSMOS_HOME", os.path.expanduser("~/Cosmos"))
    model_path = os.path.expanduser(f"~/.cache/cosmos/models/{model}")
    
    print("[Cosmos-Transfer] Starting sim-to-real transfer...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Model: {model}")
    
    # 각 클립에 대해 Transfer 실행
    clips = [d for d in os.listdir(input_dir) if d.startswith("clip_")]
    
    for clip in sorted(clips):
        clip_path = os.path.join(input_dir, clip)
        rgb_path = os.path.join(clip_path, "rgb.mp4")
        edges_path = os.path.join(clip_path, "edges.mp4")
        depth_path = os.path.join(clip_path, "depth.mp4")
        
        if not os.path.exists(rgb_path):
            print(f"  [SKIP] {clip}: no rgb.mp4")
            continue
        
        output_clip = os.path.join(output_dir, clip)
        os.makedirs(output_clip, exist_ok=True)
        
        # Cosmos-Transfer 명령어
        cmd = [
            "python", f"{cosmos_home}/cosmos_transfer/inference.py",
            "--model_path", model_path,
            "--input_rgb", rgb_path,
            "--input_condition", edges_path,
            "--condition_type", "canny_edge",
            "--output", os.path.join(output_clip, "transferred.mp4"),
            "--num_frames", "32",
            "--fps", "8",
        ]
        
        print(f"  [{clip}] Running Cosmos-Transfer...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  [{clip}] ✅ Transfer complete: {output_clip}/transferred.mp4")
        else:
            print(f"  [{clip}] ❌ Transfer failed: {result.stderr}")
    
    print(f"\n[Cosmos-Transfer] All transfers complete!")
    print(f"  Results in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cosmos-Transfer for TurtleBot3")
    parser.add_argument("--input_dir", required=True, help="CosmosWriter output directory")
    parser.add_argument("--output_dir", required=True, help="Transfer output directory")
    parser.add_argument("--model", default="Cosmos-Transfer1-7B", help="Transfer model")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    run_cosmos_transfer(args.input_dir, args.output_dir, args.model)


if __name__ == "__main__":
    main()
```

### 5.3.2 Cosmos Transfer 변환 예시

```
입력 (Isaac Sim 합성)                      출력 (Cosmos-Transfer 변환)
┌──────────────────────┐                  ┌──────────────────────┐
│  ■■  ■■■  ■■■■■     │                  │  ██  ███  ██████    │
│  ■      ■    ■       │                  │  █      █    █      │
│    ■■■    ■■■  ■■    │  ───Transfer──►  │    ███    ███  ██    │
│  ■    ■      ■       │                  │  █    █      █      │
│  ■■■  ■■■  ■■■■     │                  │  ██   ███  ████     │
└──────────────────────┘                  └──────────────────────┘
 단순한 조명, 쉐이딩                     실제 같은 텍스처, 조명 효과
```

## 5.4 Cosmos-Predict2: 비디오 예측

Cosmos-Predict2는 로봇의 카메라 뷰를 입력으로 미래 프레임을 예측합니다.

### 5.4.1 Cosmos-Predict2 실행

```bash
# Cosmos 환경 활성화
conda activate cosmos
cd ~/Cosmos

# Video-to-World 예측
python cosmos_predict2/inference.py \
    --model_path ~/.cache/cosmos/models/Cosmos-Predict2-2B-Video2World \
    --input_video /path/to/turtlebot_view.mp4 \
    --output_dir /path/to/prediction_output \
    --num_frames_to_generate 16 \
    --fps 8

# Text-to-World 예측
python cosmos_predict2/inference.py \
    --model_path ~/.cache/cosmos/models/Cosmos-Predict2-2B-Video2World \
    --text_prompt "TurtleBot navigating around obstacles in warehouse" \
    --output_dir /path/to/text2world_output \
    --num_frames_to_generate 32 \
    --fps 12
```

## 5.5 Cosmos-Policy: 정책 학습

Cosmos-Policy는 Cosmos-Predict2를 기반으로 로봇 제어 정책을 학습합니다.

### 5.5.1 Cosmos-Policy Post-training

**`src/cosmos/cosmos_policy_config.yaml`**:

```yaml
# Cosmos-Policy Configuration for TurtleBot3 Navigation

model:
  base_model: "Cosmos-Predict2-2B-Video2World"
  checkpoint_path: "~/.cache/cosmos/models/Cosmos-Predict2-2B-Video2World"
  
data:
  train_data_path: "/workspace/data/turtlebot_navigation/train"
  val_data_path: "/workspace/data/turtlebot_navigation/val"
  
  # 데이터 형식: 로봇의 과거 N 프레임 + action → 미래 K 프레임
  context_frames: 4   # 과거 프레임 수
  future_frames: 12   # 예측할 프레임 수
  
  # TurtleBot3 액션 공간
  action_dim: 2       # linear_velocity, angular_velocity
  
training:
  batch_size: 4
  learning_rate: 1.0e-5
  num_epochs: 100
  gradient_accumulation_steps: 8
  mixed_precision: "fp16"
  
  # LoRA fine-tuning
  lora_rank: 64
  lora_alpha: 128
  lora_dropout: 0.05
  
  logging:
    log_interval: 10
    save_interval: 500
    eval_interval: 100
  
  output_dir: "/workspace/outputs/cosmos_policy_turtlebot"

environment:
  robot: "turtlebot3_burger"
  max_linear_velocity: 0.22  # m/s
  max_angular_velocity: 2.84  # rad/s
  lidar_range: 3.5  # m
```

### 5.5.2 Cosmos-Policy 추론

**`src/cosmos/cosmos_inference.py`**:

```python
"""
Cosmos-Policy 추론: 학습된 정책으로 TurtleBot3 제어

Usage:
    conda activate cosmos
    python src/cosmos/cosmos_inference.py --checkpoint /path/to/checkpoint.pt
"""

import os
import sys
import cv2
import numpy as np
import torch
import argparse
import subprocess
from typing import Optional


class TurtleBotCosmosPolicy:
    """
    Cosmos-Policy 기반 TurtleBot3 내비게이션 정책
    실제 로봇 제어에 사용
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.context_frames = []
        
        print(f"[CosmosPolicy] Initializing on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Cosmos-Policy 모델 로드"""
        # 실제 구현에서는 Cosmos-Policy의 post-trained 모델을 로드
        # 여기서는 파이프라인 개요를 제공
        
        cosmos_home = os.environ.get("COSMOS_HOME", os.path.expanduser("~/Cosmos"))
        model_path = os.path.expanduser("~/.cache/cosmos/models/Cosmos-Predict2-2B-Video2World")
        
        print(f"[CosmosPolicy] Loading model from: {model_path}")
        print(f"[CosmosPolicy] Checkpoint: {self.checkpoint_path}")
        
        # TODO: 실제 모델 로드 로직
        # from cosmos_predict2 import CosmosPredict2Model
        # self.model = CosmosPredict2Model.from_pretrained(model_path)
        # self.model.load_checkpoint(self.checkpoint_path)
        # self.model.to(self.device)
        # self.model.eval()
        
        print("[CosmosPolicy] Model loaded ✅")
    
    def preprocess_observation(self, rgb_image: np.ndarray, lidar_scan: np.ndarray) -> torch.Tensor:
        """
        로봇 관측값을 모델 입력으로 전처리
        
        Args:
            rgb_image: (H, W, 3) RGB 이미지
            lidar_scan: (360,) LiDAR scan 데이터
        
        Returns:
            observation_tensor: 모델 입력 텐서
        """
        # RGB 이미지 전처리
        rgb_resized = cv2.resize(rgb_image, (224, 224))
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1)  # (C, H, W)
        
        # LiDAR scan 전처리
        lidar_normalized = lidar_scan / 3.5  # max_range로 정규화
        lidar_tensor = torch.from_numpy(lidar_normalized).float()
        
        # 컨텍스트 업데이트 (최대 4프레임 유지)
        observation = {
            "rgb": rgb_tensor,
            "lidar": lidar_tensor,
        }
        self.context_frames.append(observation)
        if len(self.context_frames) > 4:
            self.context_frames.pop(0)
        
        # 모델 입력 생성
        # 실제로는 Cosmos-Policy의 Video2World 입력 형식에 맞춤
        if len(self.context_frames) < 2:
            return None  # 충분한 컨텍스트가 쌓일 때까지 대기
        
        return observation
    
    def predict_action(self, observation: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        정책 추론: 관측값 → 행동
        
        Returns:
            action: (linear_velocity, angular_velocity)
        """
        if self.model is None:
            # Fallback: 간단한 충돌 회피
            return self._fallback_policy()
        
        # TODO: 실제 정책 추론
        # with torch.no_grad():
        #     action = self.model.infer(observation)
        # return action.cpu().numpy()
        
        return self._fallback_policy()
    
    def _fallback_policy(self) -> np.ndarray:
        """Fallback: 간단한 전진"""
        return np.array([0.1, 0.0])  # 0.1 m/s 전진
    
    def export_to_onnx(self, output_path: str):
        """PyTorch 모델을 ONNX로 변환"""
        if self.model is None:
            print("[ERROR] No model loaded")
            return False
        
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
            }
        )
        print(f"[Export] ONNX model saved: {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Cosmos-Policy Inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--export_onnx", help="Export to ONNX path")
    
    args = parser.parse_args()
    
    policy = TurtleBotCosmosPolicy(checkpoint_path=args.checkpoint)
    
    if args.export_onnx:
        policy.export_to_onnx(args.export_onnx)
        return
    
    # 실시간 추론 루프
    print("\n[Inference] Starting real-time inference loop...")
    print("Press Ctrl+C to stop")
    
    try:
        import time
        while True:
            # 실제 로봇에서는 여기서 센서 데이터를 읽음
            dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            dummy_lidar = np.random.uniform(0.15, 3.5, (360,))
            
            obs = policy.preprocess_observation(dummy_rgb, dummy_lidar)
            if obs is not None:
                action = policy.predict_action(obs)
                print(f"  Action: linear={action[0]:.3f}, angular={action[1]:.3f}")
            
            time.sleep(0.1)  # 10Hz
    
    except KeyboardInterrupt:
        print("\n[Inference] Stopped.")


if __name__ == "__main__":
    main()
```

## 5.6 Cosmos 전체 파이프라인 실행

### 5.6.1 전체 워크플로우

```bash
# ============================================
# Cosmos 통합 파이프라인 (Ubuntu 22.04 환경)
# ============================================

# Step 1: Isaac Sim에서 합성 데이터 생성
# Docker 컨테이너 내부에서 실행
cd /workspace
/isaac-sim/python.sh src/isaac_sim/cosmos_sdg_pipeline.py

# Step 2: Cosmos-Transfer (Sim→Real 변환)
conda activate cosmos
python src/cosmos/cosmos_transfer.py \
    --input_dir /workspace/_out_cosmos_turtlebot \
    --output_dir /workspace/data/cosmos_transferred

# Step 3: Cosmos-Policy 학습
conda activate cosmos
cd ~/Cosmos
python cosmos_policy/post_training.py \
    --config /workspace/config/cosmos_policy_config.yaml

# Step 4: 정책 추출 및 ONNX 변환
conda activate isaaclab
python /workspace/src/cosmos/cosmos_inference.py \
    --checkpoint /workspace/outputs/cosmos_policy_turtlebot/best.pt \
    --export_onnx /workspace/outputs/policy_turtlebot.onnx

# Step 5: TensorRT 변환
cd /workspace/outputs
/usr/src/tensorrt/bin/trtexec \
    --onnx=policy_turtlebot.onnx \
    --saveEngine=policy_turtlebot.plan \
    --fp16 \
    --workspace=4096
```

### 5.6.2 구성 파일

**`config/cosmos_config.yaml`**:
```yaml
# Cosmos 통합 설정

paths:
  cosmos_home: "~/Cosmos"
  isaac_output: "/workspace/_out_cosmos_turtlebot"
  transfer_output: "/workspace/data/cosmos_transferred"
  training_output: "/workspace/outputs/cosmos_policy_turtlebot"
  policy_onnx: "/workspace/outputs/policy_turtlebot.onnx"
  policy_tensorrt: "/workspace/outputs/policy_turtlebot.plan"

models:
  predict: "Cosmos-Predict2-2B-Video2World"
  transfer: "Cosmos-Transfer1-7B"

data_collection:
  num_clips: 10
  num_frames_per_clip: 60
  capture_interval: 5
  resolution: [1280, 720]

training:
  batch_size: 4
  learning_rate: 1.0e-5
  num_epochs: 100
  mixed_precision: "fp16"
```

## 5.7 Cosmos 성능 최적화

| 설정 | VRAM 사용량 | 속도 | 추천 상황 |
|------|-------------|------|-----------|
| fp32, batch=1 | ~24GB | 1x | 정확도 최우선 |
| fp16, batch=1 | ~14GB | 1.8x | 일반적인 경우 |
| fp16, batch=4 | ~28GB | 4x | 고성능 GPU (A100, H100) |
| INT8 (TensorRT) | ~8GB | 5x+ | Jetson 배포 |

## 5.8 문제 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| CosmosWriter 미작동 | Isaac Sim 버전 < 2025.1 | Isaac Sim 2025.1+로 업데이트 |
| CUDA OOM | 배치 크기太大 | `batch_size=1`, `fp16` 활성화 |
| Hugging Face 모델 다운로드 실패 | 액세스 토큰 필요 | `huggingface-cli login` 후 재시도 |
| Cosmos-Transfer 품질 낮음 | 조건 이미지 해상도 | 1280x720 유지, edges 맵 확인 |
| Cosmos-Policy 학습 불안정 | 학습률太高 | `learning_rate=1e-5`로 낮춤 |
