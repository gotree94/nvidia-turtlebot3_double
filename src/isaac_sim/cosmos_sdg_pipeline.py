#!/usr/bin/env python3
"""
TurtleBot3 Cosmos Data Generation Pipeline

Isaac Sim + CosmosWriter로 합성 데이터 수집

Usage (Isaac Sim python):
    /isaac-sim/python.sh src/isaac_sim/cosmos_sdg_pipeline.py
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Cosmos SDG Pipeline")
    parser.add_argument("--output_dir", default="/workspace/_out_cosmos_turtlebot",
                       help="Output directory for generated data")
    parser.add_argument("--num_clips", type=int, default=5, help="Number of clips")
    parser.add_argument("--num_frames", type=int, default=60, help="Frames per clip")
    parser.add_argument("--check", action="store_true", help="Check environment only")
    
    args = parser.parse_args()
    
    # Isaac Sim 환경 체크
    try:
        import omni
        import omni.replicator.core as rep
        print("[CosmosSDG] Running inside Isaac Sim")
    except ImportError:
        print("[CosmosSDG] This script must run inside Isaac Sim")
        print("[CosmosSDG] Use: /isaac-sim/python.sh src/isaac_sim/cosmos_sdg_pipeline.py")
        if args.check:
            return
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("TurtleBot3 Cosmos Data Collection Pipeline")
    print("=" * 60)
    
    print(f"\n[Config]")
    print(f"  Output:     {args.output_dir}")
    print(f"  Num clips:  {args.num_clips}")
    print(f"  Frames/clip: {args.num_frames}")
    print(f"  Resolution: 1280x720")
    print(f"  Modalities: RGB, Depth, Segmentation, Edges")
    
    # 실제 데이터 수집 파이프라인은 docs/05_cosmos_integration.md 참조
    print("\n[Pipeline] Setup complete. See documentation for full pipeline.")
    print("[Pipeline] Run with --check to verify environment only.\n")


if __name__ == "__main__":
    main()
