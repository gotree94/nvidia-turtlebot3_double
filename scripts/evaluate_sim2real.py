#!/usr/bin/env python3
"""
Sim-to-Real Evaluation Script

Isaac Sim (시뮬레이션) vs 실제 로봇 (Real) 주행 성능 비교 평가

Usage:
    python3 scripts/evaluate_sim2real.py --num_trials 10 --output results/sim2real_eval.json
"""

import os
import sys
import json
import time
import math
import argparse
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class TrialResult:
    """단일 시험 결과"""
    trial_id: int
    environment: str
    robot: str
    policy: str
    success: bool
    time_to_goal: float
    path_length: float
    collisions: int
    avg_linear_vel: float
    avg_angular_vel: float
    min_lidar_distance: float
    notes: str = ""


@dataclass
class EvaluationConfig:
    """평가 설정"""
    num_trials: int = 10
    max_time: float = 120.0  # 최대 시험 시간 (초)
    goal_distance_threshold: float = 0.2  # 목표 도달 판정 (m)
    collision_distance: float = 0.18  # 충돌 판정 (m)
    environment: str = "indoor_3x3"
    goals: List[tuple] = field(default_factory=lambda: [
        (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
        (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0),
        (1.5, 1.5), (-1.5, -1.5),
    ])


class Sim2RealEvaluator:
    """
    Sim-to-Real 평가기
    
    시뮬레이션(Isaac Sim)과 실제 로봇(Jetson Orin Nano)에서
    동일한 조건으로 정책을 평가하고 비교합니다.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.sim_results: List[TrialResult] = []
        self.real_results: List[TrialResult] = []
    
    def evaluate_simulation(self, policy_path: str, headless: bool = True) -> List[TrialResult]:
        """
        Isaac Sim 시뮬레이션에서 정책 평가
        
        Args:
            policy_path: 정책 파일 경로
            headless: GUI 없이 실행
        
        Returns:
            평가 결과 리스트
        """
        print("\n" + "=" * 60)
        print("Simulation Evaluation (Isaac Sim)")
        print("=" * 60)
        
        results = []
        
        # TODO: 실제 Isaac Lab 환경과 연동
        # 아래는 평가 파이프라인 템플릿
        
        for trial_idx in range(self.config.num_trials):
            goal = self.config.goals[trial_idx % len(self.config.goals)]
            print(f"\n  Trial {trial_idx+1}/{self.config.num_trials} - Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
            
            # 시뮬레이션 평가 로직 (실제 연동 필요)
            # from isaaclab_tasks import evaluate_policy
            # result = evaluate_policy(policy_path, goal)
            
            # 모의 결과 (실제 평가 시 대체)
            result = TrialResult(
                trial_id=trial_idx,
                environment=self.config.environment,
                robot="sim_isaac_sim",
                policy=policy_path,
                success=np.random.random() > 0.1,  # 90% 성공
                time_to_goal=np.random.uniform(15.0, 45.0),
                path_length=np.random.uniform(2.0, 5.0),
                collisions=np.random.randint(0, 2),
                avg_linear_vel=np.random.uniform(0.08, 0.15),
                avg_angular_vel=np.random.uniform(-0.3, 0.3),
                min_lidar_distance=np.random.uniform(0.2, 0.5),
            )
            
            results.append(result)
            self._print_trial_result(result)
        
        self.sim_results = results
        return results
    
    def evaluate_real_robot(self, policy_path: str) -> List[TrialResult]:
        """
        실제 TurtleBot3 (Jetson Orin Nano)에서 정책 평가
        
        Args:
            policy_path: 정책 파일 경로 (TensorRT .plan)
        
        Returns:
            평가 결과 리스트
        """
        print("\n" + "=" * 60)
        print("Real Robot Evaluation (Jetson Orin Nano)")
        print("=" * 60)
        
        results = []
        
        # TODO: 실제 ROS2 토픽 연동
        # 아래는 평가 파이프라인 템플릿
        
        for trial_idx in range(self.config.num_trials):
            goal = self.config.goals[trial_idx % len(self.config.goals)]
            print(f"\n  Trial {trial_idx+1}/{self.config.num_trials} - Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
            
            # 실제 로봇 평가 로직
            # 1. ros2 action send_goal /navigate_to_pose ...
            # 2. Wait for result or timeout
            # 3. Record metrics
            
            # 모의 결과 (실제 평가 시 대체)
            result = TrialResult(
                trial_id=trial_idx,
                environment=self.config.environment,
                robot="real_jetson_orin_nano",
                policy=policy_path,
                success=np.random.random() > 0.2,  # 80% 성공 (sim보다 낮음)
                time_to_goal=np.random.uniform(20.0, 60.0),
                path_length=np.random.uniform(2.5, 6.0),
                collisions=np.random.randint(0, 3),
                avg_linear_vel=np.random.uniform(0.06, 0.12),
                avg_angular_vel=np.random.uniform(-0.5, 0.5),
                min_lidar_distance=np.random.uniform(0.18, 0.4),
            )
            
            results.append(result)
            self._print_trial_result(result)
        
        self.real_results = results
        return results
    
    def _print_trial_result(self, result: TrialResult):
        """개별 시험 결과 출력"""
        status = "✅" if result.success else "❌"
        print(f"    {status} Time: {result.time_to_goal:.1f}s, "
              f"Path: {result.path_length:.2f}m, "
              f"Collisions: {result.collisions}")
    
    def compute_statistics(self, results: List[TrialResult]) -> dict:
        """
        평가 결과 통계 계산
        
        Args:
            results: 시험 결과 리스트
        
        Returns:
            통계 딕셔너리
        """
        if not results:
            return {}
        
        successes = [r for r in results if r.success]
        success_rate = len(successes) / len(results)
        
        stats = {
            "num_trials": len(results),
            "success_rate": success_rate,
            "avg_time_to_goal": np.mean([r.time_to_goal for r in successes]) if successes else 0,
            "std_time_to_goal": np.std([r.time_to_goal for r in successes]) if successes else 0,
            "avg_path_length": np.mean([r.path_length for r in successes]) if successes else 0,
            "std_path_length": np.std([r.path_length for r in successes]) if successes else 0,
            "avg_collisions": np.mean([r.collisions for r in results]),
            "collision_rate": len([r for r in results if r.collisions > 0]) / len(results),
            "avg_linear_vel": np.mean([r.avg_linear_vel for r in successes]) if successes else 0,
            "min_lidar_distance": np.min([r.min_lidar_distance for r in results]),
        }
        
        return stats
    
    def compare_and_report(self, output_path: str):
        """
        Sim vs Real 비교 보고서 생성
        
        Args:
            output_path: JSON 출력 경로
        """
        print("\n" + "=" * 60)
        print("Sim-to-Real Gap Analysis")
        print("=" * 60)
        
        sim_stats = self.compute_statistics(self.sim_results)
        real_stats = self.compute_statistics(self.real_results)
        
        # 갭 분석
        gap = {}
        for key in sim_stats:
            if key in real_stats:
                if sim_stats[key] != 0:
                    gap[key] = {
                        "sim": sim_stats[key],
                        "real": real_stats[key],
                        "gap": real_stats[key] - sim_stats[key],
                        "gap_percent": (real_stats[key] - sim_stats[key]) / sim_stats[key] * 100,
                    }
                else:
                    gap[key] = {
                        "sim": sim_stats[key],
                        "real": real_stats[key],
                        "gap": "N/A",
                        "gap_percent": "N/A",
                    }
        
        # 보고서 출력
        print(f"\n{'Metric':<25} {'Sim':<10} {'Real':<10} {'Gap':<10} {'Gap%':<10}")
        print("-" * 65)
        
        for metric, data in gap.items():
            if isinstance(data.get("gap"), str):
                print(f"{metric:<25} {data['sim']:<10.3f} {data['real']:<10.3f} {'N/A':<10} {'N/A':<10}")
            else:
                print(f"{metric:<25} {data['sim']:<10.3f} {data['real']:<10.3f} "
                      f"{data['gap']:<+10.3f} {data['gap_percent']:<+10.1f}%")
        
        # 결과 저장
        report = {
            "config": asdict(self.config),
            "sim_stats": sim_stats,
            "real_stats": real_stats,
            "gap": gap,
            "sim_trials": [asdict(r) for r in self.sim_results],
            "real_trials": [asdict(r) for r in self.real_results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved: {output_path}")
        
        # 요약
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        
        sim_sr = gap.get("success_rate", {}).get("sim", 0)
        real_sr = gap.get("success_rate", {}).get("real", 0)
        sr_gap = gap.get("success_rate", {}).get("gap_percent", 0)
        
        print(f"  Sim Success Rate:  {sim_sr:.1%}")
        print(f"  Real Success Rate: {real_sr:.1%}")
        print(f"  Sim-to-Real Gap:   {sr_gap:+.1f}%")
        
        if abs(sr_gap) < 10:
            print(f"\n  ✅ Excellent sim-to-real transfer!")
        elif abs(sr_gap) < 25:
            print(f"\n  ⚠️  Moderate gap. Consider more domain randomization.")
        else:
            print(f"\n  ❌ Large gap. Need significant domain randomization improvement.")
        
        return gap


def main():
    parser = argparse.ArgumentParser(description="Sim-to-Real Evaluation")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--output", type=str, default="results/sim2real_eval.json", help="Output path")
    parser.add_argument("--policy", type=str, default="outputs/policy/turtlebot_policy.plan", help="Policy path")
    parser.add_argument("--sim-only", action="store_true", help="Simulation only")
    parser.add_argument("--real-only", action="store_true", help="Real robot only")
    
    args = parser.parse_args()
    
    config = EvaluationConfig(num_trials=args.num_trials)
    evaluator = Sim2RealEvaluator(config)
    
    # Sim 평가
    if not args.real_only:
        evaluator.evaluate_simulation(args.policy)
    
    # Real 평가
    if not args.sim_only:
        evaluator.evaluate_real_robot(args.policy)
    
    # 비교 보고서
    evaluator.compare_and_report(args.output)


if __name__ == "__main__":
    main()
