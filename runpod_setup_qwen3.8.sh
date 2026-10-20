#!/bin/bash
# Environment for the Qwen3.8-27B pipeline (a thinking model, distilled from itself with its reasoning kept).
# Usage, on a pod started from the vLLM image (vllm/vllm-openai:v0.32.0: torch 2.13, transformers 5.19, CUDA 13, nvcc):
#   bash runpod_setup_qwen3.8.sh
# Paths default to /workspace (override with OCT_HOME); HF_TOKEN must be set (Qwen weights, LIMA).
#
# Why not runpod_setup.sh: it pins vllm 0.11 and transformers 4.57, which predate Qwen3.8 (model type qwen3_5, added in
# transformers 5.2; vLLM 0.32 serves it, with LoRA). The OpenRLHF fork pins transformers==4.57.0 but runs on 5.x, so it
# is installed without its pins.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
OCT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCT_HOME="${OCT_HOME:-/workspace}"
cd "$OCT_DIR"

command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
if [ ! -f openrlhf/setup.py ]; then
    git config submodule.openrlhf.url https://github.com/sdananya/OpenRLHF.git
    git submodule update --init openrlhf 2>/dev/null || { rm -rf openrlhf; git clone --depth 1 https://github.com/sdananya/OpenRLHF.git openrlhf; }
fi
# torch>=2.10 LR-scheduler shim in the fork: make it apply only when the attribute exists (as runpod_setup.sh does)
for f in openrlhf/openrlhf/cli/train_dpo.py openrlhf/openrlhf/cli/train_sft.py; do
    if grep -q '_orig_update_lr = _lr_sched.LRScheduler._update_lr' "$f"; then
        sed -i 's/_orig_update_lr = _lr_sched.LRScheduler._update_lr/_orig_update_lr = getattr(_lr_sched.LRScheduler, "_update_lr", None)/' "$f"
        sed -i 's/^_lr_sched.LRScheduler._update_lr = _patched_update_lr/if _orig_update_lr is not None:\n    _lr_sched.LRScheduler._update_lr = _patched_update_lr/' "$f"
    fi
done

# keep the image's torch / transformers / vllm: everything else goes in around them
pip install -q --no-input --no-deps -e ./openrlhf
pip install -q --no-input "deepspeed==0.19.6" accelerate datasets einops jsonlines loralib optimum peft tensorboard \
    torchdata torchmetrics transformers_stream_generator wandb "optree>=0.13.0" "pynvml>=12.0.0" bitsandbytes \
    flash-linear-attention
pip install -q --no-input --no-deps -e .
python3 - <<'PY'
import torch, transformers, vllm, deepspeed, openrlhf
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401  (needs transformers>=5.2)
print("torch", torch.__version__, "| transformers", transformers.__version__, "| vllm", vllm.__version__, "| deepspeed", deepspeed.__version__)
PY

# DeepSpeed builds FusedAdam on first use with nvcc. The image has nvcc but not the CUDA library headers that torch's
# CUDA headers include (cusparse.h, cusolverDn.h, cuda_runtime.h, ...), so install those first.
CUDA_DASH=$(python3 -c "import torch; print(torch.version.cuda.replace('.', '-'))")
apt-get update -qq
apt-get install -y -qq --no-install-recommends ninja-build "cuda-cudart-dev-${CUDA_DASH}" "libcusparse-dev-${CUDA_DASH}" \
    "libcusolver-dev-${CUDA_DASH}" "libcublas-dev-${CUDA_DASH}" "libcurand-dev-${CUDA_DASH}" \
    || apt-get install -y -qq --no-install-recommends ninja-build "cuda-libraries-dev-${CUDA_DASH}"
fused_adam() { python3 -c "
import torch
from deepspeed.ops.adam import FusedAdam
FusedAdam([torch.zeros(4, device='cuda', requires_grad=True)])
print('FusedAdam builds')"; }
if ! fused_adam; then
    echo "FusedAdam failed to build; installing every CUDA library header and retrying"
    apt-get install -y -qq --no-install-recommends "cuda-libraries-dev-${CUDA_DASH}"
    fused_adam
fi

mkdir -p "$OCT_HOME/models" "$OCT_HOME/loras"
cat > character/constants.py << EOF
DATA_PATH = "$OCT_DIR/data"
MODEL_PATH = "$OCT_HOME/models"
LORA_PATH = "$OCT_HOME/loras"
CONSTITUTION_PATH = "$OCT_DIR/constitutions"
EOF

# weights (with the chat template's default reasoning effort set to medium, see run_all.py) and the LIMA prompts
OCT_HOME="$OCT_HOME" OCT_DIR="$OCT_DIR" OCT_MODELS_DIR="$OCT_HOME/models" OCT_LORAS_DIR="$OCT_HOME/loras" \
    python3 run_all.py --model qwen3.8 --download-models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('GAIR/lima', repo_type='dataset', local_dir='$OCT_HOME/models/lima', allow_patterns=['*.jsonl'])"
echo "=== Qwen3.8 setup complete ==="
echo "export OCT_HOME=$OCT_HOME OCT_DIR=$OCT_DIR OCT_MODELS_DIR=$OCT_HOME/models OCT_LORAS_DIR=$OCT_HOME/loras"
echo "python3 run_data.py --stage dpo --model qwen3.8-27b --constitution lying"
echo "python3 run_all.py --model qwen3.8 --constitution lying --stage dpo --skip-upload --no-cleanup"
echo "python3 run_all.py --model qwen3.8 --constitution lying --stage fold"
echo "python3 run_data.py --stage sft --model qwen3.8-27b --constitution lying"
echo "python3 run_all.py --model qwen3.8 --constitution lying --stage sft --skip-upload --no-cleanup"
echo "python3 run_all.py --model qwen3.8 --constitution lying --stage final"
