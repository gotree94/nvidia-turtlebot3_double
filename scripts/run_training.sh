#!/bin/bash
# ============================================
# TurtleBot3 RL Training Pipeline
# ============================================
# Usage: bash scripts/run_training.sh [options]

set -e

# Default
NUM_ENVS=256
MAX_ITER=1000
HEADLESS="--headless"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --max_iter) MAX_ITER="$2"; shift 2 ;;
        --gui) HEADLESS="" ; shift ;;
        --help) echo "Usage: $0 [--num_envs 256] [--max_iter 1000] [--gui]"; exit 0 ;;
        *) shift ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/turtlebot_navigation/${TIMESTAMP}"
CKPT_DIR="checkpoints/turtlebot_navigation/${TIMESTAMP}"

mkdir -p $LOG_DIR $CKPT_DIR

echo "=============================================="
echo "TurtleBot3 Navigation Training"
echo "=============================================="
echo "  Timestamp:    $TIMESTAMP"
echo "  Num envs:     $NUM_ENVS"
echo "  Max iter:     $MAX_ITER"
echo "  Headless:     ${HEADLESS:-no}"
echo "  Log dir:      $LOG_DIR"
echo "  Checkpoints:  $CKPT_DIR"
echo "=============================================="

# Activate Isaac Lab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

# ========== Phase 1: Train ==========
echo ""
echo "[Phase 1] Training..."
python src/isaac_lab/train_turtlebot_navigation.py \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITER \
    $HEADLESS \
    2>&1 | tee "${LOG_DIR}/training.log"

# ========== Phase 2: Evaluate ==========
echo ""
echo "[Phase 2] Evaluating best policy..."
python src/isaac_lab/play_turtlebot_navigation.py \
    --checkpoint "${CKPT_DIR}/final.pt" \
    --num_episodes 50 \
    2>&1 | tee "${LOG_DIR}/evaluation.log"

# ========== Phase 3: Export ==========
echo ""
echo "[Phase 3] Exporting to ONNX..."
python -c "
import torch
import sys
sys.path.insert(0, 'src/isaac_lab')
from rsl_rl.modules import ActorCritic

OBS_DIM = 39
model = ActorCritic(OBS_DIM, OBS_DIM, 2, [256, 128, 64], [256, 128, 64])
ckpt = torch.load('${CKPT_DIR}/final.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

class PolicyNet(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
    def forward(self, obs):
        return self.actor(obs, deterministic=True)

policy = PolicyNet(model.actor)
dummy = torch.randn(1, OBS_DIM)
torch.onnx.export(policy, dummy, '${CKPT_DIR}/policy.onnx',
    opset_version=17,
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}})
print(f'ONNX: ${CKPT_DIR}/policy.onnx')
"

# ========== Phase 4: TensorRT (if on Jetson or x86 with TRT) ==========
echo ""
echo "[Phase 4] Converting to TensorRT..."
if command -v trtexec &> /dev/null; then
    /usr/src/tensorrt/bin/trtexec \
        --onnx="${CKPT_DIR}/policy.onnx" \
        --saveEngine="${CKPT_DIR}/policy.plan" \
        --fp16 \
        --workspace=4096 \
        2>&1 | tee "${LOG_DIR}/trt_conversion.log"
    echo "TensorRT engine: ${CKPT_DIR}/policy.plan"
else
    echo "trtexec not found. Skipping TensorRT conversion."
    echo "Run on Jetson Orin Nano:"
    echo "  /usr/src/tensorrt/bin/trtexec --onnx=${CKPT_DIR}/policy.onnx --saveEngine=policy.plan --fp16"
fi

echo ""
echo "=============================================="
echo "✅ Training pipeline complete!"
echo "=============================================="
echo "  Results: $CKPT_DIR/"
echo "  Logs:    $LOG_DIR/"
echo "=============================================="
ls -lh ${CKPT_DIR}/
