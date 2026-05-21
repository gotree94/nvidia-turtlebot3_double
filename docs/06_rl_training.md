# 06. Isaac Lab RL 학습 파이프라인

> **TurtleBot3 자율주행을 위한 강화학습 환경 구축 및 PPO 학습**

## 6.1 Isaac Lab 개요

Isaac Lab은 NVIDIA Isaac Sim 기반의 경량 로봇 학습 프레임워크입니다.

### 핵심 특징
- **GPU 가속 벡터화 환경**: 수백 개의 병렬 환경에서 동시 학습
- **Domain Randomization**: sim-to-real 격차 해소
- **다중 로봇 지원**: 단일에서 다중 로봇까지
- **다중 알고리즘**: PPO, SAC, DDPG, On-policy, Off-policy

### 학습 파이프라인 개요

```
┌────────────────────────────────────────────────────────────────────┐
│                        Isaac Lab Pipeline                           │
│                                                                     │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐        │
│  │ Manager │    │  Task    │    │  Agent  │    │  RSL    │        │
│  │ Based   │───►│  Class   │───►│  Runner │───►│  RL     │        │
│  │ Env     │    │          │    │         │    │  Engine │        │
│  └─────────┘    └──────────┘    └─────────┘    └─────────┘        │
│       │              │              │              │               │
│       │              ▼              │              │               │
│       │     ┌──────────────┐       │              │               │
│       │     │ Observations │       │              │               │
│       │     │ Actions      │       │              │               │
│       │     │ Rewards      │       │              │               │
│       │     │ Resets       │       │              │               │
│       │     └──────────────┘       │              │               │
│       │                            │              │               │
│       ▼                            ▼              ▼               │
│  ┌──────────────────────────────────────────────────────┐         │
│  │              Isaac Sim (PhysX)                       │         │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │         │
│  │  │ Env #1   │ │ Env #2   │ │ Env #N   │ ...         │         │
│  │  │ TurtleBot│ │ TurtleBot│ │ TurtleBot│             │         │
│  │  └──────────┘ └──────────┘ └──────────┘             │         │
│  └──────────────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────────────┘
```

## 6.2 TurtleBot3 Navigation Task 정의

**`src/isaac_lab/turtlebot_navigation_cfg.py`**:

```python
"""
TurtleBot3 Navigation - Isaac Lab 환경 설정

Isaac Lab의 Manager 기반 환경 설정.
TurtleBot3 차동 구동 로봇의 장애물 회피 내비게이션 태스크.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
import torch

# ===================================================
# 로봇 설정
# ===================================================
@dataclass
class TurtleBot3RobotCfg:
    """TurtleBot3 Burger 로봇 물리 설정"""
    
    # USD 파일 경로
    usd_path: str = "/workspace/src/urdf/turtlebot3_burger.usd"
    
    # 초기 상태
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.02)
    initial_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    
    # 물리 파라미터
    wheel_radius: float = 0.033
    wheel_separation: float = 0.160
    max_linear_velocity: float = 0.22
    max_angular_velocity: float = 2.84
    
    # 조인트 이름
    joint_names: List[str] = field(default_factory=lambda: [
        "wheel_left_joint",
        "wheel_right_joint",
    ])
    
    # DOF 이름
    dof_names: List[str] = field(default_factory=lambda: [
        "wheel_left_joint",
        "wheel_right_joint",
    ])
    
    # DOF 속성
    dof_velocity_limit: float = 10.0
    dof_effort_limit: float = 1.0


# ===================================================
# 관측값 설정
# ===================================================
@dataclass
class ObservationsCfg:
    """관측값 공간 설정"""
    
    # LiDAR scan 설정
    lidar_num_rays: int = 360
    lidar_min_range: float = 0.15
    lidar_max_range: float = 3.5
    lidar_horizontal_fov: float = 360.0
    
    # Downsampled LiDAR (학습 효율을 위해 축소)
    lidar_downsampled: int = 36  # 360 → 36 (10° 간격)
    
    # 기타 관측값
    use_goal_observation: bool = True
    use_velocity_observation: bool = True
    use_orientation_observation: bool = True
    
    def get_observation_dim(self) -> int:
        """관측값 차원 계산"""
        dim = self.lidar_downsampled  # LiDAR
        if self.use_goal_observation:
            dim += 2  # goal (dx, dy)
        if self.use_velocity_observation:
            dim += 2  # linear, angular velocity
        if self.use_orientation_observation:
            dim += 1  # heading error
        return dim


# ===================================================
# 행동 공간 설정
# ===================================================
@dataclass
class ActionsCfg:
    """행동 공간 설정 (차동 구동)"""
    
    # 차동 구동: [linear_velocity, angular_velocity]
    action_dim: int = 2
    
    # 행동 스케일
    linear_velocity_scale: float = 0.22  # max linear velocity
    angular_velocity_scale: float = 2.84  # max angular velocity
    
    # 행동 제한
    min_linear_velocity: float = -0.10  # 후진 허용
    max_linear_velocity: float = 0.22
    min_angular_velocity: float = -2.84
    max_angular_velocity: float = 2.84


# ===================================================
# 보상 함수 설정
# ===================================================
@dataclass
class RewardsCfg:
    """보상 함수 가중치 설정"""
    
    # 메인 보상
    goal_reward_weight: float = 10.0       # 목표 도달
    progress_weight: float = 5.0            # 목표 접근 진행률
    
    # 패널티
    collision_weight: float = -10.0         # 충돌 패널티
    angular_velocity_weight: float = -0.1  # 회전 패널티 (부드러운 주행 유도)
    linear_velocity_weight: float = 0.1    # 전진 보상 (빠른 주행 유도)
    
    # 선택 보상
    goal_heading_weight: float = 1.0        # 목표 방향 정렬
    smoothness_weight: float = -0.05        # 행동 변화율 패널티
    
    # 목표 조건
    goal_distance_threshold: float = 0.2    # 목표 도달 판정 거리 (m)
    collision_distance_threshold: float = 0.18  # 충돌 판정 거리 (m)


# ===================================================
# 환경 설정
# ===================================================
@dataclass
class EnvCfg:
    """환경 설정"""
    
    # 환경 크기
    env_length: float = 8.0    # x축 길이 (m)
    env_width: float = 8.0     # y축 길이 (m)
    
    # 장애물
    num_obstacles: int = 5
    obstacle_min_size: float = 0.2
    obstacle_max_size: float = 0.5
    
    # 목표
    goal_reset_prob: float = 0.3  # 에피소드마다 목표 재설정 확률
    
    # 타임아웃
    max_episode_length: int = 256  # 최대 스텝 수


# ===================================================
# Domain Randomization 설정
# ===================================================
@dataclass
class DomainRandomizationCfg:
    """Domain Randomization 설정 (Sim-to-Real)"""
    
    # 물리 파라미터 랜덤화
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.3, 1.5)
    
    randomize_mass: bool = True
    mass_range: Tuple[float, float] = (0.7, 1.3)  # 배율
    
    randomize_motor_strength: bool = True
    motor_strength_range: Tuple[float, float] = (0.8, 1.2)
    
    # 센서 노이즈
    randomize_lidar_noise: bool = True
    lidar_noise_std: float = 0.02  # 2cm noise
    
    randomize_odometry_drift: bool = True
    odometry_drift_std: float = 0.01  # 1% drift
    
    # 환경 랜덤화
    randomize_lighting: bool = True
    randomize_obstacle_positions: bool = True


# ===================================================
# 전체 설정 통합
# ===================================================
@dataclass
class TurtleBot3NavigationCfg:
    """TurtleBot3 Navigation 전체 설정"""
    
    # 환경 이름
    task_name: str = "TurtleBot3Navigation"
    
    # 하위 설정
    robot: TurtleBot3RobotCfg = field(default_factory=TurtleBot3RobotCfg)
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    domain_randomization: DomainRandomizationCfg = field(default_factory=DomainRandomizationCfg)
    
    # 학습 설정
    num_envs: int = 256       # 병렬 환경 수
    num_obs: int = 0          # 자동 계산됨
    num_actions: int = 2      # [linear_vel, angular_vel]
    num_steps: int = 256      # PPO rollout steps
    seed: int = 42
    
    def __post_init__(self):
        self.num_obs = self.observations.get_observation_dim()
```

## 6.3 Task Class 구현

**`src/isaac_lab/turtlebot_navigation_task.py`**:

```python
"""
TurtleBot3 Navigation Task Class

Isaac Lab의 ManagerBasedRLEnv를 상속받아 TurtleBot3 내비게이션 태스크 구현.
"""

import torch
import numpy as np
import math
import random
from typing import Tuple, Dict, Optional, List

# Isaac Lab imports
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    CurriculumTermCfg as CurriTerm,
)
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import RayCaster, Camera
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from .turtlebot_navigation_cfg import TurtleBot3NavigationCfg


class TurtleBot3NavigationEnv(ManagerBasedRLEnv):
    """
    TurtleBot3 Navigation 강화학습 환경
    
    이 환경은 TurtleBot3 Burger 로봇이 LiDAR와 Odometry 정보를 사용하여
    장애물을 피하면서 목표 지점까지 이동하는 태스크를 정의합니다.
    """
    
    def __init__(self, cfg: TurtleBot3NavigationCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        self._cfg = cfg
        self._goal_positions = None
        self._obstacle_positions = None
        self._prev_dist_to_goal = None
        self._steps = 0
        
        print(f"[TurtleBot3Navigation] Initialized with {cfg.num_envs} environments")
        print(f"[TurtleBot3Navigation] Observation dim: {cfg.num_obs}")
        print(f"[TurtleBot3Navigation] Action dim: {cfg.num_actions}")
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        관측값 수집
        
        Returns:
            dict: 관측값 텐서
        """
        obs = {}
        
        # 1. LiDAR scan 데이터 (360 → 36 다운샘플)
        lidar_data = self._sensors["lidar"].data.ray_hits  # (num_envs, 360)
        lidar_downsampled = lidar_data.view(
            lidar_data.shape[0], 36, 10
        ).mean(dim=-1)  # (num_envs, 36)
        obs["lidar"] = lidar_downsampled / self._cfg.observations.lidar_max_range
        
        # 2. 목표 상대 위치
        robot_positions = self._robot.data.root_pos_w  # (num_envs, 3)
        goal_relative = self._goal_positions - robot_positions[:, :2]
        obs["goal"] = goal_relative / self._cfg.env.env_length  # 정규화
        
        # 3. 현재 속도
        linear_vel = self._robot.data.root_lin_vel_w[:, :2].norm(dim=-1, keepdim=True)
        angular_vel = self._robot.data.root_ang_vel_w[:, 2:3]
        obs["velocity"] = torch.cat([linear_vel, angular_vel], dim=-1)
        
        # 4. Heading error (목표 방향과 현재 방향의 차이)
        robot_yaw = self._get_yaw_from_quat(self._robot.data.root_quat_w)
        goal_angle = torch.atan2(goal_relative[:, 1], goal_relative[:, 0])
        heading_error = self._wrap_angle(goal_angle - robot_yaw)
        obs["heading_error"] = heading_error.unsqueeze(-1)
        
        # 정책 관측값 결합
        policy_obs = torch.cat([
            obs["lidar"],
            obs["goal"],
            obs["velocity"],
            obs["heading_error"],
        ], dim=-1)
        
        return {"policy": policy_obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        보상 계산
        
        보상 구성:
        1. 목표 도달 보상 (+10.0)
        2. 진행 보상 (목표 거리 감소량 +5.0)
        3. 충돌 패널티 (-10.0)
        4. 회전 패널티 (-0.1 * |angular_vel|)
        5. 전진 보상 (+0.1 * linear_vel)
        6. 방향 정렬 보상 (+1.0 * cosine(heading_error))
        7. 행동 부드러움 패널티
        
        Returns:
            rewards: (num_envs,) 보상 텐서
        """
        cfg = self._cfg.rewards
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 로봇 위치
        robot_pos = self._robot.data.root_pos_w  # (num_envs, 3)
        
        # 목표까지 거리
        dist_to_goal = torch.norm(
            robot_pos[:, :2] - self._goal_positions, dim=-1
        )
        
        # 1. 목표 도달 보상
        goal_reached = dist_to_goal < cfg.goal_distance_threshold
        rewards[goal_reached] += cfg.goal_reward_weight
        
        # 2. 진행 보상 (거리 감소)
        if self._prev_dist_to_goal is not None:
            progress = self._prev_dist_to_goal - dist_to_goal
            rewards += progress * cfg.progress_weight
        self._prev_dist_to_goal = dist_to_goal.clone()
        
        # 3. 충돌 패널티
        lidar_min = self._sensors["lidar"].data.ray_hits.min(dim=-1)[0]
        collision = lidar_min < cfg.collision_distance_threshold
        rewards[collision] += cfg.collision_weight
        
        # 4. 회전 패널티
        angular_vel = self._robot.data.root_ang_vel_w[:, 2]
        rewards += torch.abs(angular_vel) * cfg.angular_velocity_weight
        
        # 5. 전진 보상
        linear_vel = self._robot.data.root_lin_vel_w[:, :2].norm(dim=-1)
        rewards += linear_vel * cfg.linear_velocity_weight
        
        # 6. 방향 정렬 보상
        robot_yaw = self._get_yaw_from_quat(self._robot.data.root_quat_w)
        goal_angle = torch.atan2(
            self._goal_positions[:, 1] - robot_pos[:, 1],
            self._goal_positions[:, 0] - robot_pos[:, 0],
        )
        heading_error = self._wrap_angle(goal_angle - robot_yaw)
        heading_alignment = torch.cos(heading_error)
        rewards += heading_alignment * cfg.goal_heading_weight
        
        # 7. 행동 부드러움 (선택)
        if hasattr(self, '_prev_actions'):
            action_change = torch.norm(
                self._actions - self._prev_actions, dim=-1
            )
            rewards += action_change * cfg.smoothness_weight
        self._prev_actions = self._actions.clone()
        
        return rewards
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        종료 조건 확인
        
        Returns:
            dones: (num_envs,) 에피소드 종료 플래그
            time_outs: (num_envs,) 타임아웃 플래그
        """
        robot_pos = self._robot.data.root_pos_w
        dist_to_goal = torch.norm(robot_pos[:, :2] - self._goal_positions, dim=-1)
        
        # 목표 도달
        goal_reached = dist_to_goal < self._cfg.rewards.goal_distance_threshold
        
        # 충돌
        lidar_min = self._sensors["lidar"].data.ray_hits.min(dim=-1)[0]
        collision = lidar_min < self._cfg.rewards.collision_distance_threshold
        
        # 타임아웃
        time_out = self._steps >= self._cfg.env.max_episode_length
        
        dones = goal_reached | collision | time_out
        
        return dones, time_out.expand_as(dones)
    
    def _reset_envs(self, env_ids: torch.Tensor):
        """
        환경 초기화
        
        Args:
            env_ids: 초기화할 환경 인덱스
        """
        # 로봇 위치 초기화
        num_resets = len(env_ids)
        
        # 랜덤 초기 위치 (환경 경계 내)
        init_positions = torch.zeros((num_resets, 3), device=self.device)
        init_positions[:, 0] = torch.rand(num_resets, device=self.device) * 4.0 - 2.0
        init_positions[:, 1] = torch.rand(num_resets, device=self.device) * 4.0 - 2.0
        init_positions[:, 2] = 0.02
        
        # 랜덤 초기 방향
        init_orientations = torch.zeros((num_resets, 4), device=self.device)
        init_orientations[:, 0] = 1.0
        init_orientations[:, 3] = torch.rand(num_resets, device=self.device) * 2 * math.pi
        
        self._robot.write_root_pose_to_sim(
            torch.cat([init_positions, init_orientations], dim=-1), env_ids
        )
        
        # 목표 위치 초기화
        goal_positions = torch.zeros((num_resets, 2), device=self.device)
        goal_positions[:, 0] = torch.rand(num_resets, device=self.device) * 6.0 - 3.0
        goal_positions[:, 1] = torch.rand(num_resets, device=self.device) * 6.0 - 3.0
        
        self._goal_positions[env_ids] = goal_positions
        
        # 거리 추적 초기화
        if self._prev_dist_to_goal is not None:
            robot_pos = self._robot.data.root_pos_w[env_ids]
            dist = torch.norm(robot_pos[:, :2] - goal_positions, dim=-1)
            self._prev_dist_to_goal[env_ids] = dist
    
    def _setup_scene(self):
        """씬 구성"""
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.assets import ArticationCfg, RigidObjectCfg
        from isaaclab.sensors import RayCasterCfg, CameraCfg
        from isaaclab.sim import PhysxCfg
        
        # 로봇 에셋
        from isaaclab.assets import ArticulationCfg
        self._robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/TurtleBot3",
            spawn=UsdFileCfg(
                usd_path=self._cfg.robot.usd_path,
                rigid_props=ArticulationCfg.RigidProperties(
                    max_linear_velocity=self._cfg.robot.max_linear_velocity,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self._cfg.robot.initial_position,
                rot=self._cfg.robot.initial_orientation,
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "wheels": ImplicitActuatorCfg(
                    joint_names_expr=[".*_joint"],
                    effort_limit=self._cfg.robot.dof_effort_limit,
                    velocity_limit=self._cfg.robot.dof_velocity_limit,
                ),
            },
        )
        
        # LiDAR 센서
        self._lidar_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/TurtleBot3/base_scan",
            offset=RayCasterCfg.OffsetCfg(pos=(-0.032, 0.0, 0.077)),
            num_rays=self._cfg.observations.lidar_num_rays,
            depth_range=(self._cfg.observations.lidar_min_range, 
                         self._cfg.observations.lidar_max_range),
            horizontal_fov=self._cfg.observations.lidar_horizontal_fov,
        )
        
        self.scene = InteractiveScene(
            InteractiveSceneCfg(
                num_envs=self._cfg.num_envs,
                env_spacing=10.0,
                replicate_physics=True,
            )
        )
    
    # 유틸리티 함수
    @staticmethod
    def _get_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
        """Quaternion → Yaw 각도 변환"""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)
    
    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        """각도를 [-π, π] 범위로 래핑"""
        return (angle + math.pi) % (2 * math.pi) - math.pi


# ===================================================
# Isaac Lab Task 등록
# ===================================================
from isaaclab_tasks import register_task

@register_task(
    "TurtleBot3Navigation-v0",
    description="TurtleBot3 Burger navigation with LiDAR in Isaac Lab",
)
def register_turtlebot_navigation():
    from .turtlebot_navigation_cfg import TurtleBot3NavigationCfg
    return TurtleBot3NavigationCfg, TurtleBot3NavigationEnv
```

## 6.4 학습 실행 스크립트

**`src/isaac_lab/train_turtlebot_navigation.py`**:

```python
"""
TurtleBot3 Navigation PPO 학습 실행 스크립트

Usage:
    conda activate isaaclab
    python src/isaac_lab/train_turtlebot_navigation.py --num_envs 256 --headless
"""

import os
import sys
import argparse
import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.utils import update_class_from_dict
from isaaclab_tasks import register_tasks
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.utils import store_code_state


def train_turtlebot_navigation(args):
    """TurtleBot3 Navigation 학습 실행"""
    
    print("=" * 60)
    print("TurtleBot3 Navigation - PPO Training")
    print("=" * 60)
    print(f"  Num envs: {args.num_envs}")
    print(f"  Headless: {args.headless}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Device: {args.device}")
    print("=" * 60)
    
    # 환경 설정 로드
    env_cfg = parse_env_cfg(
        task_name="TurtleBot3Navigation-v0",
        use_gpu=torch.cuda.is_available(),
        num_envs=args.num_envs,
    )
    
    # 학습 파라미터 설정
    env_cfg.num_steps = 256
    env_cfg.ppo.learning_rate = 3.0e-4
    env_cfg.ppo.num_minibatches = 4
    env_cfg.ppo.update_epochs = 5
    env_cfg.ppo.gamma = 0.99
    env_cfg.ppo.lam = 0.95
    env_cfg.ppo.clip_param = 0.2
    env_cfg.ppo.value_loss_coef = 1.0
    env_cfg.ppo.entropy_coef = 0.01
    env_cfg.ppo.max_grad_norm = 1.0
    
    # 환경 생성
    env = gym.make("TurtleBot3Navigation-v0", cfg=env_cfg)
    
    # PPO Runner 설정
    runner = OnPolicyRunner(
        env=env,
        device=args.device,
        num_steps=env_cfg.num_steps,
        log_dir=f"logs/turtlebot_navigation/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # Actor-Critic 네트워크
    runner.alg.actor_critic = ActorCritic(
        num_actor_obs=env_cfg.num_obs,
        num_critic_obs=env_cfg.num_obs,
        num_actions=env_cfg.num_actions,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    ).to(args.device)
    
    # 학습 시작
    print("\n[Training] Starting PPO training...")
    
    for iteration in range(args.max_iterations):
        # Rollout 수집
        obs = env.reset()
        for step in range(env_cfg.num_steps):
            actions = runner.alg.actor_critic.act(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            runner.alg.process_env_step(rewards, dones, infos)
            obs = next_obs
        
        # PPO 업데이트
        train_info = runner.alg.update()
        
        # 로깅
        if iteration % 10 == 0:
            avg_reward = train_info.get("avg_reward", 0)
            success_rate = train_info.get("success_rate", 0)
            print(f"  Iter {iteration:4d}: reward={avg_reward:.2f}, "
                  f"success={success_rate:.2%}, "
                  f"lr={runner.alg.optimizer.param_groups[0]['lr']:.2e}")
        
        # 체크포인트 저장
        if iteration % 100 == 0 and iteration > 0:
            runner.save(f"checkpoints/turtlebot_navigation/iter_{iteration:06d}.pt")
            print(f"  [Checkpoint] Saved at iteration {iteration}")
    
    # 최종 모델 저장
    runner.save("checkpoints/turtlebot_navigation/final.pt")
    print(f"\n✅ Training complete!")
    print(f"   Model saved: checkpoints/turtlebot_navigation/final.pt")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="TurtleBot3 Navigation Training")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training")
    
    args = parser.parse_args()
    train_turtlebot_navigation(args)


if __name__ == "__main__":
    main()
```

## 6.5 학습된 정책 평가

**`src/isaac_lab/play_turtlebot_navigation.py`**:

```python
"""
학습된 TurtleBot3 Navigation 정책 평가 스크립트

Usage:
    conda activate isaaclab
    python src/isaac_lab/play_turtlebot_navigation.py --checkpoint checkpoints/final.pt
"""

import os
import sys
import torch
import argparse
import gymnasium as gym
import numpy as np
from isaaclab_tasks.utils import parse_env_cfg


def evaluate_policy(args):
    """학습된 정책 평가"""
    
    print("=" * 60)
    print("TurtleBot3 Navigation - Policy Evaluation")
    print("=" * 60)
    
    # 환경 설정
    env_cfg = parse_env_cfg(
        task_name="TurtleBot3Navigation-v0",
        use_gpu=torch.cuda.is_available(),
        num_envs=1,  # 단일 환경 평가
    )
    
    # 환경 생성 (렌더링 활성화)
    env = gym.make("TurtleBot3Navigation-v0", cfg=env_cfg)
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Actor-Critic 네트워크 복원
    from rsl_rl.modules import ActorCritic
    actor_critic = ActorCritic(
        num_actor_obs=env_cfg.num_obs,
        num_critic_obs=env_cfg.num_obs,
        num_actions=env_cfg.num_actions,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
    ).to(args.device)
    
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    
    print(f"  Model loaded from: {args.checkpoint}")
    
    # 평가
    num_episodes = args.num_episodes
    total_rewards = []
    success_count = 0
    collision_count = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 1000:
            with torch.no_grad():
                action = actor_critic.act(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
        
        total_rewards.append(episode_reward)
        
        if info.get("goal_reached", False):
            success_count += 1
        if info.get("collision", False):
            collision_count += 1
        
        print(f"  Episode {ep+1:3d}: reward={episode_reward:.1f}, "
              f"steps={step:3d}, success={info.get('goal_reached', False)}")
    
    # 결과 통계
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = success_count / num_episodes * 100
    collision_rate = collision_count / num_episodes * 100
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Average Reward: {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"  Success Rate:   {success_rate:.1f}%")
    print(f"  Collision Rate: {collision_rate:.1f}%")
    print(f"  Episodes:       {num_episodes}")
    print("=" * 60)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate TurtleBot3 Navigation Policy")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of test episodes")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    evaluate_policy(args)


if __name__ == "__main__":
    main()
```

## 6.6 Policy Export: PyTorch → ONNX → TensorRT

**`scripts/export_policy.sh`**:

```bash
#!/bin/bash
# 학습된 정책을 ONNX로 변환 후 TensorRT 최적화
# Usage: bash scripts/export_policy.sh

set -e

# 설정
CHECKPOINT_DIR="checkpoints/turtlebot_navigation"
OUTPUT_DIR="outputs/policy"
ONNX_MODEL="${OUTPUT_DIR}/turtlebot_policy.onnx"
TRT_MODEL="${OUTPUT_DIR}/turtlebot_policy.plan"
OBS_DIM=39  # 36 (lidar) + 2 (goal) + 1 (heading error)

mkdir -p $OUTPUT_DIR

echo "=== Step 1: PyTorch → ONNX ==="
conda activate isaaclab
python -c "
import torch
import sys
sys.path.insert(0, 'src/isaac_lab')

from rsl_rl.modules import ActorCritic

# 모델 로드
model = ActorCritic(
    num_actor_obs=$OBS_DIM,
    num_critic_obs=$OBS_DIM,
    num_actions=2,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
)
checkpoint = torch.load('${CHECKPOINT_DIR}/final.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Actor만 추출 (정책 추론용)
class PolicyNetwork(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
    
    def forward(self, obs):
        return self.actor(obs, deterministic=True)

policy = PolicyNetwork(model.actor)

# ONNX Export
dummy_input = torch.randn(1, $OBS_DIM)
torch.onnx.export(
    policy,
    dummy_input,
    '$ONNX_MODEL',
    opset_version=17,
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={
        'observation': {0: 'batch_size'},
        'action': {0: 'batch_size'},
    }
)
print(f'ONNX exported: $ONNX_MODEL')
"

echo ""
echo "=== Step 2: ONNX → TensorRT ==="
/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_MODEL \
    --saveEngine=$TRT_MODEL \
    --fp16 \
    --workspace=4096 \
    --minShapes=observation:1x$OBS_DIM \
    --optShapes=observation:1x$OBS_DIM \
    --maxShapes=observation:4x$OBS_DIM

echo ""
echo "=== Export Complete ==="
echo "  ONNX:     $ONNX_MODEL"
echo "  TensorRT: $TRT_MODEL"
ls -lh $ONNX_MODEL $TRT_MODEL
```

## 6.7 PPO 하이퍼파라미터 요약

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `num_envs` | 256 | 병렬 환경 수 |
| `num_steps` | 256 | Rollout 수집 스텝 |
| `learning_rate` | 3.0e-4 | Adam 학습률 |
| `num_minibatches` | 4 | 미니배치 수 |
| `update_epochs` | 5 | Epoch 수 |
| `gamma` | 0.99 | 할인율 |
| `lam` | 0.95 | GAE lambda |
| `clip_param` | 0.2 | PPO 클리핑 |
| `entropy_coef` | 0.01 | 엔트로피 계수 |
| `value_loss_coef` | 1.0 | 가치 손실 계수 |
| `max_grad_norm` | 1.0 | 그래디언트 클리핑 |
| `actor_hidden_dims` | [256, 128, 64] | Actor 네트워크 구조 |
| `critic_hidden_dims` | [256, 128, 64] | Critic 네트워크 구조 |
| `activation` | elu | 활성화 함수 |

## 6.8 보상 함수 요약

| 보상 항목 | 계수 | 설명 |
|-----------|------|------|
| Goal reached | +10.0 | 목표 도달 즉시 |
| Progress | +5.0 × Δdistance | 목표 접근 시 |
| Collision | -10.0 | 장애물 충돌 시 |
| Angular vel | -0.1 × \|ω\| | 회전 시 패널티 |
| Linear vel | +0.1 × v | 전진 시 보상 |
| Heading alignment | +1.0 × cos(θ_error) | 목표 방향 정렬 |
| Smoothness | -0.05 × \|Δa\| | 행동 급변 패널티 |

## 6.9 실행 명령어 요약

```bash
# 학습
conda activate isaaclab
python src/isaac_lab/train_turtlebot_navigation.py \
    --num_envs 256 --max_iterations 1000 --headless

# TensorBoard 모니터링
tensorboard --logdir logs/turtlebot_navigation

# 평가
python src/isaac_lab/play_turtlebot_navigation.py \
    --checkpoint checkpoints/turtlebot_navigation/final.pt \
    --num_episodes 50

# ONNX / TensorRT 변환
bash scripts/export_policy.sh
```

## 6.10 문제 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| 학습 수렴 안 됨 | 보상 함수 불균형 | 보상 가중치 조정, entropy_coef 증가 |
| 로봇이 제자리 회전만 함 | LiDAR 관측 문제 | `lidar_downsampled` 간격 확인 |
| GPU 메모리 부족 | num_envs 너무 큼 | `--num_envs 64`로 감소 |
| Collision 너무 잦음 | 장애물 밀도 높음 | `num_obstacles` 감소 |
| Sim-to-Real 실패 | Domain Randomization 부족 | friction/mass/noise 범위 확장 |
