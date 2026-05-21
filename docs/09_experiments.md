# 09. 실험 및 테스트 가이드

> **단계별 실험 설계, 평가 방법론, 문제 해결**

## 9.1 실험 로드맵

```
Phase 1 ── 기본 검증: Isaac Sim + ROS2 통신 + 수동 주행
    │
Phase 2 ── Nav2 통합: SLAM + 경로 계획 + 자율 주행
    │
Phase 3 ── RL 학습: Isaac Lab PPO 학습 + Policy 평가
    │
Phase 4 ── Cosmos: 합성 데이터 생성 + Transfer + Policy
    │
Phase 5 ── Sim-to-Real: Policy → Jetson 배포 + 실제 주행
    │
Phase 6 ── 듀얼 로봇: 대형 제어 + 협업 내비게이션
```

## 9.2 Phase 1: 기본 검증 실험

### 9.2.1 Isaac Sim + ROS2 통신 테스트

**목표**: Isaac Sim에서 TurtleBot3가 ROS2 메시지를 주고받는지 확인

```bash
# ========== 실험 절차 ==========

# 1. Isaac Sim 실행 (Docker)
cd docker
bash isaac_sim_docker.sh
# 컨테이너 내부에서 Isaac Sim GUI 실행
./isaac-sim.sh

# 2. ROS2 Bridge 활성화
# Isaac Sim → Window → ROS2 → Enable ROS2 Bridge

# 3. URDF Import → TurtleBot3 USD 로드
# Isaac Utils → URDF Importer → turtlebot3_burger.urdf

# 4. ROS2 토픽 확인 (터미널)
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
ros2 topic list
# 예상 출력:
#   /cmd_vel
#   /odom
#   /scan
#   /joint_states
#   /tf
#   /clock

# 5. LiDAR 데이터 확인
ros2 topic echo /scan --once

# 6. Teleop 주행 테스트
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# 키보드 i/j/l/로 로봇 움직임 확인
```

**Pass Criteria**:
- [ ] `ros2 topic list`에 `/odom`, `/scan`, `/cmd_vel` 표시
- [ ] LiDAR scan에 유효한 범위 값 (0.15 ~ 3.5m)
- [ ] Teleop 키보드로 로봇 움직임 제어 가능
- [ ] `/odom`의 position이 주행에 따라 변경

### 9.2.2 차동 구동 정밀도 테스트

```bash
# ========== 실험 절차 ==========

# 1. 직진 1m 테스트
ros2 topic pub --once /cmd_vel geometry_msgs/Twist \
    '{linear: {x: 0.15}, angular: {z: 0.0}}'
sleep 6.7  # 1m / 0.15m/s
ros2 topic pub --once /cmd_vel geometry_msgs/Twist \
    '{linear: {x: 0.0}, angular: {z: 0.0}}'

# 2. 회전 90° 테스트
ros2 topic pub --once /cmd_vel geometry_msgs/Twist \
    '{linear: {x: 0.0}, angular: {z: 0.5}}'
sleep 3.14  # (π/2) / 0.5
ros2 topic pub --once /cmd_vel geometry_msgs/Twist \
    '{linear: {x: 0.0}, angular: {z: 0.0}}'

# 3. 결과 기록
ros2 bag record /odom /scan /cmd_vel -o test_drive_basic
```

**Pass Criteria**:
- [ ] 직진 1m 오차 < 5cm
- [ ] 90° 회전 오차 < 3°
- [ ] 직진 시 Y축 편차 < 2cm
- [ ] Bag record 정상 저장

## 9.3 Phase 2: Nav2 통합 테스트

### 9.3.1 SLAM 테스트

```bash
# ========== 실험 절차 ==========

# Terminal 1: Isaac Sim + TurtleBot3 실행
# Terminal 2: SLAM 실행
ros2 launch slam_toolbox async_slam_toolbox_node.py

# Terminal 3: Teleop으로 맵 작성
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# 환경 전체를 천천히 탐색하며 맵 생성 (약 5분)

# Terminal 4: 맵 저장
ros2 run nav2_map_server map_saver_cli -f maps/test_map
```

**Pass Criteria**:
- [ ] 점유 맵이 정상적으로 생성됨
- [ ] 장애물이 맵에 올바르게 표시
- [ ] 루프 클로징 정상 동작
- [ ] 맵 저장 완료 (test_map.pgm + test_map.yaml)

### 9.3.2 Nav2 자율 주행 테스트

```yaml
# config/nav2_test_params.yaml (Phase 2 테스트용)
controller_server:
  ros__parameters:
    controller_frequency: 20.0
    FollowPath:
      desired_linear_vel: 0.15  # 느린 속도로 시작
      max_linear_accel: 0.10
      lookahead_dist: 0.3
      use_rotate_to_heading: true
```

```bash
# ========== 실험 절차 ==========

# Terminal 1: Isaac Sim + TurtleBot3
# Terminal 2: Nav2 실행
ros2 launch nav2_bringup navigation_launch.py \
    params_file:=config/nav2_test_params.yaml

# Terminal 3: RViz에서 Nav2 Goal 설정
rviz2

# RViz에서 "Nav2 Goal" 버튼 클릭 → 맵 상의 목표 지점 지정
```

**Pass Criteria**:
- [ ] Global path가 장애물을 회피하여 생성됨
- [ ] 로봇이 경로를 따라 목표까지 도달
- [ ] 동적 장애물 추가 시 경로 재계획
- [ ] 목표 도달 시 정지

## 9.4 Phase 3: RL 학습 실험

### 9.4.1 학습 실험 설계

| 실험 ID | 환경 수 | Hidden Dim | 학습률 | 보상 구조 | 설명 |
|---------|---------|------------|--------|-----------|------|
| EXP-01 | 64 | [128, 64] | 3e-4 | 기본 | 기준 실험 |
| EXP-02 | 256 | [256, 128, 64] | 3e-4 | 기본 | 대규모 병렬 |
| EXP-03 | 256 | [256, 128, 64] | 1e-3 | 기본 | 높은 학습률 |
| EXP-04 | 256 | [256, 128, 64] | 3e-4 | +Heading | 방향 정렬 추가 |
| EXP-05 | 256 | [256, 128, 64] | 3e-4 | +Smoothness | 부드러움 추가 |
| EXP-06 | 256 | [256, 128, 64] | 3e-4 | +DR | Domain Randomization |

### 9.4.2 학습 실행

```bash
# EXP-01: Baseline
conda activate isaaclab
python src/isaac_lab/train_turtlebot_navigation.py \
    --num_envs 64 --max_iterations 500 --headless

# EXP-02: 대규모 병렬 (RTX 4090 권장)
python src/isaac_lab/train_turtlebot_navigation.py \
    --num_envs 256 --max_iterations 1000 --headless

# TensorBoard 모니터링
tensorboard --logdir logs/turtlebot_navigation --bind_all
```

### 9.4.3 학습 평가 메트릭

| 메트릭 | 측정 방법 | 목표 값 |
|--------|-----------|---------|
| Success Rate | 100 에피소드 중 목표 도달 비율 | > 90% |
| Average Return | 에피소드당 평균 보상 | > 500 |
| Average Path Length | 목표까지 이동 거리 | < 1.5x 직선 거리 |
| Collision Rate | 충돌 발생 에피소드 비율 | < 5% |
| Steps to Goal | 평균 도달 스텝 수 | < 200 |
| Angular Velocity Variance | 회전 명령 분산 | < 0.5 |

### 9.4.4 학습 곡면 분석

```
Reward Trend (기대)
  ▲
  │
  │         ╱╲     ╱╲
  │        ╱  ╲   ╱  ╲
  │       ╱    ╲ ╱    ╲
  │   ╱╲╱            ╲
  │  ╱                  ╲
  │ ╱                    ╲
  └─────────────────────────► Iteration
  0    200   400   600   800  1000
  
  초기: 탐험 위주 (보상 낮음)
  중기: 학습 가속 (보상 급증)
  후기: 수렴 (보상 안정화)
```

## 9.5 Phase 4: Cosmos 실험

### 9.5.1 합성 데이터 품질 평가

```bash
# ========== 데이터 수집 ==========
conda activate isaaclab
/isaac-sim/python.sh src/isaac_sim/cosmos_sdg_pipeline.py
# 출력: _out_cosmos_turtlebot/clip_0000~0004/

# ========== 데이터 검증 ==========
# 각 클립의 프레임 수 확인
for clip in _out_cosmos_turtlebot/clip_*/; do
    echo "$clip: $(ls "$clip"/*.mp4 | wc -l) modalities"
done

# Cosmos-Transfer 실행
conda activate cosmos
python src/cosmos/cosmos_transfer.py \
    --input_dir _out_cosmos_turtlebot \
    --output_dir data/cosmos_transferred
```

**Pass Criteria**:
- [ ] 5개 클립 × 5개 모달리티 = 25개 mp4 파일 생성
- [ ] 각 클립 60프레임 이상
- [ ] Cosmos-Transfer 출력물이 육안으로 현실적
- [ ] Segmentation mask가 올바른 클래스 할당

### 9.5.2 Cosmos-Policy 실험

| 실험 | 데이터셋 | 모델 | Epochs | 성능 |
|------|---------|------|--------|------|
| CP-01 | 5 clips | Cosmos-Predict2-2B | 50 | Baseline |
| CP-02 | 10 clips | Cosmos-Predict2-2B | 100 | +Data |
| CP-03 | 10 clips + Transfer | Cosmos-Predict2-2B | 100 | +Transfer |

## 9.6 Phase 5: Sim-to-Real 테스트

### 9.6.1 Policy 변환 테스트

```bash
# PyTorch → ONNX 변환 확인
python -c "
import onnx
model = onnx.load('outputs/policy/turtlebot_policy.onnx')
onnx.checker.check_model(model)
print(f'ONNX model verified: {model.graph.input[0].name} → {model.graph.output[0].name}')
print(f'Input shape: {model.graph.input[0].type.tensor_type.shape}')
"

# TensorRT 변환 확인
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=outputs/policy/turtlebot_policy.plan \
    --warmUp=100 \
    --duration=10

# 예상 출력:
#   Throughput: xxx fps
#   Latency: xx ms
```

### 9.6.2 실제 로봇 주행 실험

```bash
# ========== 실험 설정 ==========
# 실내 3m × 3m 공간에 장애물 배치
# 목표 지점까지 5회 주행

# ========== 실행 ==========
# Jetson Orin Nano에서
export ROS_DOMAIN_ID=42
export TURTLEBOT3_MODEL=burger

# TurtleBot3 bringup
ros2 launch turtlebot3_bringup turtlebot3_robot.launch.py &

# Policy inference
python3 src/deployment/jetson_inference_node.py \
    --model_path outputs/policy/turtlebot_policy.plan &
```

### 9.6.3 Sim-to-Real 평가 결과 기록

```bash
# 평가 스크립트
python3 scripts/evaluate_sim2real.py \
    --num_trials 10 \
    --output results/sim2real_eval.json

# 결과 샘플
cat results/sim2real_eval.json
{
    "sim_success_rate": 0.95,
    "real_success_rate": 0.80,
    "sim_avg_steps": 145,
    "real_avg_steps": 178,
    "sim_collision_rate": 0.03,
    "real_collision_rate": 0.10,
    "gap_success": 0.15,
    "gap_steps": 33
}
```

**평가 양식**:
```yaml
# results/trial_log.yaml
trial_001:
  date: "2026-05-22"
  robot: "jetson_orin_nano"
  policy: "exp_02_iter_1000"
  environment: "실내 3x3m, 장애물 4개"
  
  metrics:
    success: true
    time_to_goal: 34.2s
    path_length: 4.7m
    collisions: 1
    avg_linear_vel: 0.12 m/s
  
  observations:
    - "LiDAR 노이즈로 인한 불안정한 회전"
    - "조명 변화에 강건함"
    - "카펫에서 미끄러짐 발생"
  
  improvements:
    - "Domain Randomization에 카펫 마찰 추가 필요"
    - "LiDAR 노이즈 필터링 강화"
```

## 9.7 Phase 6: 듀얼 로봇 실험

### 9.7.1 Formation Control 실험

| 대형 | Separation | 선속도 | 성공률 | 평균 오차 |
|------|-----------|--------|--------|-----------|
| Column | 0.5m | 0.1 m/s | 100% | 0.03m |
| Column | 0.8m | 0.1 m/s | 95% | 0.05m |
| Line | 0.8m | 0.1 m/s | 90% | 0.08m |
| Diamond | 1.0m | 0.08 m/s | 85% | 0.10m |

### 9.7.2 듀얼 로봇 협업 테스트

```bash
# ========== 테스트 시나리오 ==========

# Scenario A: 동시 출발 → 각자 목표
ros2 action send_goal /tb1/navigate_to_pose nav2_msgs/action/NavigateToPose \
    '{pose: {header: {frame_id: "map"}, pose: {position: {x: 2.0, y: 1.0}, orientation: {w: 1.0}}}}'
ros2 action send_goal /tb2/navigate_to_pose nav2_msgs/action/NavigateToPose \
    '{pose: {header: {frame_id: "map"}, pose: {position: {x: -2.0, y: -1.0}, orientation: {w: 1.0}}}}'

# Scenario B: Leader-Follower Column
python3 src/deployment/formation_controller.py \
    --formation column --separation 0.8

# Scenario C: 충돌 회피 (교차 경로)
python3 src/deployment/formation_controller.py \
    --formation staggered --separation 0.6
```

## 9.8 성능 벤치마크

### 9.8.1 추론 성능

| 플랫폼 | 정책 모델 | 추론 시간 | 전력 | FPS |
|--------|-----------|-----------|------|-----|
| RTX 4090 | Policy (256,128,64) | 0.5ms | 350W | 2000 |
| Jetson Orin Nano | ONNX FP32 | 15ms | 15W | 66 |
| Jetson Orin Nano | TensorRT FP16 | 5ms | 15W | 200 |
| Jetson Orin Nano | TensorRT INT8 | 3ms | 15W | 333 |
| Raspberry Pi 5 | Policy (CPU) | 150ms | 5W | 6 |

### 9.8.2 네트워크 지연

| 연결 방식 | 평균 RTT | 패킷 손실 | ROS2 안정성 |
|-----------|---------|-----------|-------------|
| Ethernet (유선) | 0.5ms | < 0.1% | ⭐⭐⭐⭐⭐ |
| WiFi 6 (5GHz) | 3ms | < 0.5% | ⭐⭐⭐⭐ |
| WiFi 5 (2.4GHz) | 15ms | < 3% | ⭐⭐⭐ |

## 9.9 문제 해결 체크리스트

### 9.9.1 Isaac Sim 문제

```
□ Isaac Sim GUI가 안 열림
  → xhost +local:docker 실행
  → DISPLAY 환경변수 :0 또는 :1 확인
  → Docker 볼륨 마운트 확인

□ URDF Import 실패
  → Fix Base Link 체크 해제
  → Joint Drive Type = Velocity
  → USD 파일 경로 한글/공백 없음

□ ROS2 Bridge 안 됨
  → Window → ROS2 → Enable ROS2 Bridge
  → ROS_DOMAIN_ID 일치
  → RMW_IMPLEMENTATION 일치
```

### 9.9.2 Isaac Lab 문제

```
□ 학습이 수렴하지 않음
  → 보상 함수 스케일링 확인
  → entropy_coef 조정 (0.01 → 0.05)
  → learning_rate 조정 (3e-4 → 1e-3)

□ GPU OOM
  → num_envs 감소 (256 → 128 → 64)
  → Hidden dim 축소 ([256,128,64] → [128,64])
  → 배치 크기 감소

□ 로봇이 움직이지 않음
  → USD 파일 경로 확인
  → Joint drive 설정 확인
  → 컨트롤러 연결 확인
```

### 9.9.3 Jetson 배포 문제

```
□ 모델 로드 실패
  → ONNX/TensorRT 파일 존재 확인
  → CUDA 버전 호환성 확인
  → `onnxruntime-gpu` 설치 확인

□ 추론 속도 느림
  → GPU 사용 확인 (tegrastats)
  → TensorRT FP16 변환 확인
  → MAXN 모드 설정

□ ROS2 통신 두절
  → Domain ID 확인
  → 방화벽(iptables) 확인
  → CycloneDDS XML 설정 확인
```

## 9.10 결과 보고 템플릿

```markdown
# 실험 보고서

## 실험 정보
- **실험 ID**: EXP-XX
- **날짜**: 2026-05-22
- **실험자**: 
- **목표**: 

## 환경 설정
- **Isaac Sim 버전**: 2025.2.0
- **Isaac Lab 버전**: 2.1.0
- **Cosmos 버전**: 2.0
- **Jetson**: Orin Nano (JetPack 6.0)
- **RPi**: Raspberry Pi 5

## 실험 결과

### 메트릭
| 메트릭 | 값 | 목표 | 달성 여부 |
|--------|-----|------|-----------|
| Success Rate | | > 90% | |
| Avg Return | | > 500 | |
| Collision Rate | | < 5% | |

### 관찰 사항
1. 
2. 
3. 

### 개선 제안
1. 
2. 
3. 

## 결론
```

## 9.11 전체 프로젝트 체크리스트

- [ ] **Phase 1**: Isaac Sim + ROS2 기본 통신 성공
- [ ] **Phase 1**: URDF → USD 변환 성공
- [ ] **Phase 1**: LiDAR 센서 ROS2 publish 확인
- [ ] **Phase 1**: Teleop 키보드 주행 확인
- [ ] **Phase 2**: SLAM 맵 생성 성공
- [ ] **Phase 2**: Nav2 자율 주행 성공
- [ ] **Phase 2**: 동적 장애물 회피 확인
- [ ] **Phase 3**: Isaac Lab 학습 환경 정상 동작
- [ ] **Phase 3**: PPO 학습 수렴
- [ ] **Phase 3**: Policy 평가 (Success Rate > 90%)
- [ ] **Phase 4**: CosmosWriter 데이터 수집 성공
- [ ] **Phase 4**: Cosmos-Transfer 변환 성공
- [ ] **Phase 4**: Cosmos-Policy 학습 (선택)
- [ ] **Phase 5**: ONNX 변환 성공
- [ ] **Phase 5**: TensorRT 변환 성공
- [ ] **Phase 5**: Jetson 실시간 추론 성공
- [ ] **Phase 5**: 실제 로봇 자율 주행 성공
- [ ] **Phase 6**: RPi에 ROS2 Jazzy + TurtleBot3 설치
- [ ] **Phase 6**: Dual ROS2 통신 확인
- [ ] **Phase 6**: Formation control 성공
