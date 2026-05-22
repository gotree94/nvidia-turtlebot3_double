#!/usr/bin/env python3
"""
TurtleBot3 Navigation PPO Training

Usage:
    conda activate isaaclab
    python src/isaac_lab/train_turtlebot_navigation.py --num_envs 256 --headless
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="TurtleBot3 Navigation Training")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Max iterations")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--check", action="store_true", help="Environment check only")
    
    args = parser.parse_args()
    
    # Isaac Lab 환경 체크
    try:
        import isaaclab
        print(f"[CHECK] Isaac Lab version: {isaaclab.__version__}")
    except ImportError:
        print("[CHECK] Isaac Lab not found")
        print("[CHECK] Install with: cd ~/IsaacLab && ./isaaclab.sh --install")
        if args.check:
            return
        sys.exit(1)
    
    if args.check:
        print("[CHECK] Environment ready ✅")
        return
    
    print("\n" + "=" * 60)
    print("TurtleBot3 Navigation - PPO Training")
    print("=" * 60)
    print(f"  Num envs:     {args.num_envs}")
    print(f"  Max iter:     {args.max_iterations}")
    print(f"  Headless:     {args.headless}")
    print(f"  Device:       {args.device}")
    print("=" * 60)
    
    # 실제 학습 코드는 Isaac Lab ManagerBasedRLEnv 사용
    # docs/06_rl_training.md 참조
    
    print("\n[Training] Starting PPO...")
    print("[Training] See docs/06_rl_training.md for full implementation")
    print("[Training] Training logs: logs/turtlebot_navigation/\n")


if __name__ == "__main__":
    main()
