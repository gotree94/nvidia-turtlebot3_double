# NVIDIA TurtleBot3 Dual Autonomous Navigation Project

> **NVIDIA Isaac Sim + Cosmos + Isaac Lab 기반 TurtleBot3 자율주행 풀스택 프로젝트**

![Project Status](https://img.shields.io/badge/status-development-yellow)
![NVIDIA Isaac Sim](https://img.shields.io/badge/Isaac_Sim-2025.2-green)
![NVIDIA Cosmos](https://img.shields.io/badge/Cosmos-2.0-blue)
![ROS2](https://img.shields.io/badge/ROS2-Humble/Jazzy-orange)

## 📋 프로젝트 개요

이 프로젝트는 **NVIDIA 최신 로보틱스 AI 기술 스택**을 활용하여 TurtleBot3 기반의 자율주행 시스템을 구축하는 종합 레퍼런스 아키텍처입니다.

### 핵심 기술 스택

| 기술 | 버전 | 역할 |
|------|------|------|
| **NVIDIA Isaac Sim** | 2025.2+ | 물리 기반 로봇 시뮬레이션 (Omniverse) |
| **NVIDIA Isaac Lab** | 2.1+ | 강화학습(RL) 트레이닝 프레임워크 |
| **NVIDIA Cosmos** | 2.0 | World Foundation Model, 합성 데이터 생성, Policy 학습 |
| **NVIDIA Jetson Orin Nano** | JetPack 6+ | 실시간 추론 및 엣지 AI 배포 |
| **ROS2** | Humble / Jazzy | 로봇 미들웨어 및 분산 통신 |
| **Navigation2 (Nav2)** | 최신 | 경로 계획 및 제어 |

### 프로젝트 확장 구조

```
Phase 1 ─── Single TurtleBot3 (Jetson Orin Nano) + Isaac Sim
    │
Phase 2 ─── Cosmos 합성 데이터 + Isaac Lab RL 학습
    │
Phase 3 ─── Sim-to-Real: 학습된 Policy → 실제 로봇 배포
    │
Phase 4 ─── Dual Robot: RPi TurtleBot3 + Jetson Orin Nano TurtleBot3 협업
```

## 📁 프로젝트 구조

```
nvidia-turtlebos3_double/
├── README.md                     # 프로젝트 개요 (this file)
├── docs/                         # 전체 문서
│   ├── 01_prerequisites.md       # 사전 요구사항 및 아키텍처 개요
│   ├── 02_environment_setup.md   # 환경 설정 가이드
│   ├── 03_urdf_modeling.md       # URDF 로봇 모델링
│   ├── 04_isaac_sim.md           # Isaac Sim 시뮬레이션
│   ├── 05_cosmos_integration.md  # Cosmos 통합 및 합성 데이터
│   ├── 06_rl_training.md         # Isaac Lab RL 학습 파이프라인
│   ├── 07_inference.md           # 추론 및 Jetson 배포
│   ├── 08_dual_robot.md          # Dual TurtleBot3 확장
│   └── 09_experiments.md         # 실험 및 테스트 가이드
├── src/                          # 소스 코드
│   ├── urdf/                     # URDF 로봇 모델 파일
│   ├── isaac_sim/                # Isaac Sim 관련 스크립트
│   ├── isaac_lab/                # Isaac Lab 학습 환경
│   ├── deployment/               # 실제 로봇 배포 코드
│   └── cosmos/                   # Cosmos 파이프라인
├── config/                       # 설정 파일
├── docker/                       # Docker 컨테이너 설정
├── scripts/                      # 셸 스크립트 (설치/실행)
└── assets/                       # 추가 리소스
```

## 🚀 빠른 시작

### 1. 시스템 요구사항

| 구성 요소 | 최소 사양 | 권장 사양 |
|-----------|-----------|-----------|
| **GPU** | NVIDIA RTX 3060 (12GB) | NVIDIA RTX 4090 (24GB) 이상 |
| **CPU** | Intel i7-12xxx / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| **RAM** | 32GB | 64GB |
| **Storage** | 100GB | 500GB (SSD) |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| **Docker** | 24+ | 24+ |

### 2. 설치 개요

```bash
# 저장소 클론
git clone <this-repo> nvidia-turtlebos3_double
cd nvidia-turtlebos3_double

# 호스트 환경 설정
bash scripts/setup_host.sh

# Docker 기반 Isaac Sim + ROS2 환경
cd docker
docker compose -f docker-compose.yaml build
```

### 3. 문서 로드맵

| 문서 | 설명 | 난이도 |
|------|------|--------|
| [01. 사전 요구사항](docs/01_prerequisites.md) | 시스템 구성, 아키텍처 개요, 의존성 | ⭐ |
| [02. 환경 설정](docs/02_environment_setup.md) | Isaac Sim, ROS2, Cosmos, Jetson 설치 | ⭐⭐ |
| [03. URDF 모델링](docs/03_urdf_modeling.md) | TurtleBot3 URDF 구조, 커스텀 모델 | ⭐⭐ |
| [04. Isaac Sim](docs/04_isaac_sim.md) | 시뮬레이션 환경, 센서, ROS2 브릿지 | ⭐⭐⭐ |
| [05. Cosmos 통합](docs/05_cosmos_integration.md) | 합성 데이터 생성, Cosmos Transfer | ⭐⭐⭐ |
| [06. RL 학습](docs/06_rl_training.md) | Isaac Lab PPO 학습, 보상 함수 | ⭐⭐⭐⭐ |
| [07. 추론 및 배포](docs/07_inference.md) | Jetson Orin Nano 실시간 추론 | ⭐⭐⭐ |
| [08. Dual Robot](docs/08_dual_robot.md) | 다중 로봇 시스템 확장 | ⭐⭐⭐⭐ |
| [09. 실험 가이드](docs/09_experiments.md) | 테스트, 평가, 문제 해결 | ⭐⭐ |

## 🧠 기술 상세

### NVIDIA 기술 스택 통합 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                        DGX / Desktop                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │   Isaac Sim     │  │  Isaac Lab   │  │    Cosmos Suite    │  │
│  │  (Simulation)   │◄─┤(RL Training) │  │  (WFM + Transfer)  │  │
│  │                 │  │              │  │                    │  │
│  │ • URDF → USD    │  │ • PPO/SAC    │  │ • 합성 데이터 생성 │  │
│  │ • 센서 시뮬레이션 │  │ • 보상 함수  │  │ • Video2World     │  │
│  │ • ROS2 Bridge   │  │ • Domain Rand│  │ • Policy 후처리   │  │
│  └────────┬────────┘  └──────┬───────┘  └─────────┬──────────┘  │
│           │                  │                     │             │
│           └──────────────────┼─────────────────────┘             │
│                              │ Policy Export                     │
└──────────────────────────────┼──────────────────────────────────┘
                               │ .onnx / .pt
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     AGX / Jetson Orin Nano                       │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  ROS2 Node (C++/Python)                                │      │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │      │
│  │  │ LiDAR    │  │  Camera  │  │  Policy Inference   │   │      │
│  │  │ Driver   │  │  Driver  │  │  (TensorRT)         │   │      │
│  │  └────┬─────┘  └────┬─────┘  └──────────┬─────────┘   │      │
│  │       │             │                   │              │      │
│  │       ▼             ▼                   ▼              │      │
│  │  ┌──────────────────────────────────────────────┐      │      │
│  │  │  Navigation2 (Nav2) Stack                    │      │      │
│  │  │  • Global Planner • Local Planner • BT       │      │      │
│  │  └──────────────────────────────────────────────┘      │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  Robot 2: Raspberry Pi 5 TurtleBot3 (ROS2 Agent)       │      │
│  │  • Nav2 경량 실행   • 센서 데이터 퓨전   • 명령 수신    │      │
│  └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### 주요 학습 파이프라인

```
[Isaac Sim 환경] ──RGB/Depth/LiDAR──► [Cosmos Transfer] ─► [합성 데이터 증강]
       │                                                           │
       │                                                           ▼
       │                                              [Cosmos Policy / Isaac Lab]
       │                                                      │
       │                                                      ▼
       └──────── ←── [Rollout] ────────── [RL 학습 (PPO)] ───┘
                                                      │
                                                      ▼
                                            [Policy Export: .onnx]
                                                      │
                                                      ▼
                                         [TensorRT 변환 → Jetson]
                                                      │
                                                      ▼
                                         [실시간 추론 → /cmd_vel]
```

## 🔬 주요 특징

1. **Cosmos 기반 합성 데이터 증강**: Isaac Sim에서 생성한 데이터를 Cosmos Transfer로 현실감 있게 변환
2. **Isaac Lab PPO 학습**: GPU 가속 벡터화 환경에서 대규모 병렬 RL 학습
3. **Sim-to-Real Zero-shot 전이**: 시뮬레이션에서 학습한 Policy를 실제 로봇에 직접 배포
4. **Dual Robot 협업**: RPi + Jetson Orin Nano 이기종 로봇 시스템 간 협업 내비게이션
5. **TensorRT 최적화**: Jetson Orin Nano에서 실시간 추론을 위한 최적화

## 📚 참고 자료

- [NVIDIA Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [NVIDIA Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [NVIDIA Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/)
- [NVIDIA Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)
- [TurtleBot3 ROS2 Tutorials](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)
- [Navigation2 Documentation](https://navigation.ros.org/)

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다. 각 구성 요소(NVIDIA, ROS2, TurtleBot3)의 라이선스를 따릅니다.

---

*이 프로젝트는 NVIDIA의 최신 Physical AI 기술 스택을 기반으로 구축되었습니다.*
