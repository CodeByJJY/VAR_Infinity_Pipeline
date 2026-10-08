#!/bin/bash
#SBATCH -p amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -J infinity2b_kv_probe
#SBATCH -o /scratch/e2032a10/Infinity/profiles/a100_2b_kv_probe_%j.out
#SBATCH -e /scratch/e2032a10/Infinity/profiles/a100_2b_kv_probe_%j.err

set -euo pipefail

cd /scratch/e2032a10/Infinity

source /scratch/e2032a10/Infinity/.venv_inf/bin/activate

export PYTHONPATH=/scratch/e2032a10/Infinity:/scratch/e2032a10/Infinity/tools
export HF_HOME=/scratch/e2032a10/.cache/huggingface
export HF_HUB_CACHE=/scratch/e2032a10/.cache/huggingface/hub
export TMPDIR=/scratch/e2032a10/tmp
export PIP_CACHE_DIR=/scratch/e2032a10/.cache/pip

echo "===== JOB INFO ====="
date
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch=',torch.__version__,'cuda=',torch.version.cuda,'gpu=',torch.cuda.get_device_name(0))"

echo "===== 8B REPEAT START ====="

python -u tools/probe_real_kv.py \
  --pn 1M \
  --model_path weights/infinity_2b_reg.pth \
  --vae_type 32 \
  --vae_path weights/infinity_vae_d32reg.pth \
  --model_type infinity_2b \
  --text_encoder_ckpt weights/flan-t5-xl \
  --apply_spatial_patchify 0 \
  --use_flex_attn 0 \
  --checkpoint_type torch \
  --bf16 1 \
  --cfg 3 \
  --tau 0.5 \
  --prompt "a photo of a red panda sitting on a wooden chair" \
  --batch_size 2 \
  --save_file outputs/a100_2b_kv_probe.jpg

echo "===== 8B REPEAT END ====="
date
