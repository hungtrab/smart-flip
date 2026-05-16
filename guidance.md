# Colab Guidance

This guide covers three current tasks:

- RTN vs RTN+Flip, 3-bit, in the main repo.
- RTN vs RTN+Flip, 4-bit, in the main repo.
- Llama3 FlatQuant + EGBC tune, in the isolated Llama3 repo.

The RTN tasks should use the PPL-first strategy: run cheap perplexity first, then run lm-eval only for the best 2-3 configs.

## Paths

Use these paths on Colab unless you intentionally changed them:

```bash
MAIN_REPO=/content/smart-flip
LLAMA3_REPO=/content/llama3/smart-flip
```

Main repo is for:

```text
RTN 3-bit
RTN 4-bit
Qwen / Mistral / Llama3.1
```

Llama3 repo is for:

```text
meta-llama/Meta-Llama-3-8B FlatQuant + EGBC tune
```

## Environment Variables

Create `.env` in each repo if needed:

```bash
HF_TOKEN=<your_hf_token>
HUGGINGFACE_HUB_TOKEN=<your_hf_token>
WANDB_API_KEY=<your_wandb_key>
```

Do not use plain `pip`; use `python -m pip` inside the active venv.

## Setup Main Repo For RTN Tasks

Use this for RTN 3-bit and RTN 4-bit:

```python
!bash -lc 'cd /content/smart-flip && \
python3 -m venv .venv && \
source .venv/bin/activate && \
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
python /tmp/get-pip.py && \
python -m pip install --upgrade pip setuptools wheel && \
python -m pip install -r requirements.txt && \
python - <<PY
import sys, torch, transformers, datasets, wandb
print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("wandb", wandb.__version__)
PY'
```

If the venv already exists:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && python -m pip install -r requirements.txt'
```

Files needed for the current RTN scripts:

```text
main.py
src/quantization/pipeline.py
scripts/bash/rtn_flip_b3_ppl_first.sh
scripts/bash/rtn_flip_b3.sh
scripts/bash/rtn_flip_b4.sh
scripts/bash/rtn_flip_b4_ppl_first.sh
```

Important: RTN scripts now default to `FLATQUANT_A_BITS=16`, so the RTN check is weight-only by default instead of W3A4/W4A4.

## Setup Llama3 Repo

Preferred setup for the isolated repo:

```python
!bash -lc 'cd /content/llama3/smart-flip && \
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi && \
uv venv --system-site-packages --python python3 .venv && \
source .venv/bin/activate && \
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
python /tmp/get-pip.py && \
python -m pip install --upgrade pip setuptools wheel && \
python -m pip install --no-cache-dir numpy safetensors sentencepiece protobuf tiktoken pyarrow accelerate peft tqdm psutil wandb python-dotenv evaluate scikit-learn rouge-score sacrebleu sqlitedict pytablewriter word2number more-itertools && \
python -m pip install --no-cache-dir --force-reinstall --no-deps "transformers==4.36.0" "tokenizers==0.15.2" "datasets==2.17.1" "lm-eval==0.4.4" "huggingface-hub==0.24.6" "fsspec==2023.10.0" "dill==0.3.8" "multiprocess==0.70.16" "xxhash" && \
python - <<PY
import sys, torch, transformers, datasets, lm_eval, wandb
print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("lm_eval", getattr(lm_eval, "__version__", "unknown"))
print("wandb", wandb.__version__)
PY'
```

If you have the local helper copied into the repo, this is shorter:

```python
!bash -lc 'cd /content/smart-flip && LLAMA3_REPO_DIR=/content/llama3/smart-flip bash colab/setup_llama3_full_pipeline_env.sh'
```

Files that matter most for Llama3 compatibility:

```text
flatquant/model_tools/llama_utils.py
flatquant/data_utils.py
src/evaluation/flatquant_data_utils.py
src/evaluation/lm_eval.py
scripts/bash/llama3_flatquant_b3_tune_egbc.sh
```

## Task 1: RTN 3-Bit

Run PPL sweep for all default models:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b3_ppl_first \
LOG_DIR=./logs/rtn_flip_b3_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b3-ppl-first \
GPU=0 \
BITS=3 \
STAGE=ppl \
C4_SAMPLES=128 \
SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b3_ppl_first.sh'
```

Run PPL for one model:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b3_ppl_first \
LOG_DIR=./logs/rtn_flip_b3_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b3-ppl-first \
GPU=0 BITS=3 STAGE=ppl C4_SAMPLES=128 SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b3_ppl_first.sh'
```

Other model values:

```text
mistralai/Mistral-7B-v0.3
meta-llama/Meta-Llama-3.1-8B
Qwen/Qwen2.5-7B
```

Continue PPL after Colab reconnect:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b3_ppl_first \
LOG_DIR=./logs/rtn_flip_b3_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b3-ppl-first \
GPU=0 BITS=3 STAGE=ppl C4_SAMPLES=128 SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b3_ppl_first.sh'
```

This skips completed JSON files under:

```text
/content/smart-flip/results/eval/
```

Run lm-eval only for top 3 Flip configs by C4 PPL:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b3_ppl_first \
LOG_DIR=./logs/rtn_flip_b3_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b3-ppl-first \
GPU=0 BITS=3 STAGE=lm_eval_top TOP_N=3 SELECT_BY=C4 SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b3_ppl_first.sh'
```

Run lm-eval top 3 for one model:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
MODEL_PATH=Qwen/Qwen2.5-7B \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b3_ppl_first \
LOG_DIR=./logs/rtn_flip_b3_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b3-ppl-first \
GPU=0 BITS=3 STAGE=lm_eval_top TOP_N=3 SELECT_BY=C4 SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b3_ppl_first.sh'
```

## Task 2: RTN 4-Bit

Use the dedicated 4-bit PPL-first script: `scripts/bash/rtn_flip_b4_ppl_first.sh`.

Run PPL sweep for all default models:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b4_ppl_first \
LOG_DIR=./logs/rtn_flip_b4_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b4-ppl-first \
GPU=0 \
BITS=4 \
STAGE=ppl \
C4_SAMPLES=128 \
SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b4_ppl_first.sh'
```

Run PPL for one model:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b4_ppl_first \
LOG_DIR=./logs/rtn_flip_b4_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b4-ppl-first \
GPU=0 BITS=4 STAGE=ppl C4_SAMPLES=128 SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b4_ppl_first.sh'
```

Continue PPL:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b4_ppl_first \
LOG_DIR=./logs/rtn_flip_b4_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b4-ppl-first \
GPU=0 BITS=4 STAGE=ppl C4_SAMPLES=128 SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b4_ppl_first.sh'
```

Run lm-eval top 3:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b4_ppl_first \
LOG_DIR=./logs/rtn_flip_b4_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b4-ppl-first \
GPU=0 BITS=4 STAGE=lm_eval_top TOP_N=3 SELECT_BY=C4 SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b4_ppl_first.sh'
```

## Task 3: Llama3 EGBC Tune

This runs:

```text
KNEE_VALUES=(0.0 0.01 0.02 0.03 0.04 0.05)
MAX_FLIP_VALUES=(0.01 0.02 0.03 0.04 0.05)
```

Run from scratch or continue automatically:

```python
!bash -lc 'cd /content/llama3/smart-flip && source .venv/bin/activate && \
export HF_DATASETS_CACHE=/content/llama3/smart-flip/data/cache/hf_datasets/datasets-llama3-b3 && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
RUN_RAW=auto \
PYTHON_BIN=/content/llama3/smart-flip/.venv/bin/python \
WANDB_PROJECT=egbc_tune_llama3_b3 \
GPU=0 \
SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/llama3_flatquant_b3_tune_egbc.sh'
```

Continue after raw artifact is already generated:

```python
!bash -lc 'cd /content/llama3/smart-flip && source .venv/bin/activate && \
export HF_DATASETS_CACHE=/content/llama3/smart-flip/data/cache/hf_datasets/datasets-llama3-b3 && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
RUN_RAW=0 \
PYTHON_BIN=/content/llama3/smart-flip/.venv/bin/python \
WANDB_PROJECT=egbc_tune_llama3_b3 \
GPU=0 \
SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/llama3_flatquant_b3_tune_egbc.sh'
```

Continue from a specific grid point:

```python
!bash -lc 'cd /content/llama3/smart-flip && source .venv/bin/activate && \
export HF_DATASETS_CACHE=/content/llama3/smart-flip/data/cache/hf_datasets/datasets-llama3-b3 && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
RUN_RAW=0 \
START_KNEE=0.03 \
START_FLIP=0.04 \
PYTHON_BIN=/content/llama3/smart-flip/.venv/bin/python \
WANDB_PROJECT=egbc_tune_llama3_b3 \
GPU=0 \
SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/llama3_flatquant_b3_tune_egbc.sh'
```

Exact string matters for resume:

```text
START_KNEE=0.0, not 0
START_FLIP=0.05, not 0.050
```

Force rerun the whole Llama3 tune using existing raw artifact:

```python
!bash -lc 'cd /content/llama3/smart-flip && source .venv/bin/activate && \
export HF_DATASETS_CACHE=/content/llama3/smart-flip/data/cache/hf_datasets/datasets-llama3-b3 && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE START_KNEE START_FLIP && \
RUN_RAW=0 \
SKIP_EXISTING_JSON=0 \
PYTHON_BIN=/content/llama3/smart-flip/.venv/bin/python \
WANDB_PROJECT=egbc_tune_llama3_b3 \
GPU=0 \
stdbuf -oL -eL bash scripts/bash/llama3_flatquant_b3_tune_egbc.sh'
```

## Backup For Continue

RTN tasks only need JSON results to continue:

```python
!tar -czf /content/rtn_resume_backup.tar.gz \
  /content/smart-flip/results/eval/ \
  /content/smart-flip/logs/rtn_flip_b3_ppl_first/ \
  /content/smart-flip/logs/rtn_flip_b4_ppl_first/ \
  /content/smart-flip/logs/rtn_flip_b4/ \
  /content/smart-flip/logs/rtn_flip_b3/ 2>/dev/null || true
```

Llama3 tune needs raw FlatQuant params plus JSON results:

```python
!mkdir -p /content/backup/flatquant_raw_Meta-Llama-3-8B_b3
!cp /content/llama3/smart-flip/results/models/flatquant_raw/flatquant_raw_Meta-Llama-3-8B_b3/flat_parameters.pth /content/backup/flatquant_raw_Meta-Llama-3-8B_b3/
!cp /content/llama3/smart-flip/results/models/flatquant_raw/flatquant_raw_Meta-Llama-3-8B_b3/metadata.json /content/backup/flatquant_raw_Meta-Llama-3-8B_b3/
!cp -r /content/llama3/smart-flip/results/eval /content/backup/eval_llama3
!tar -czf /content/llama3_resume_minimal.tar.gz -C /content backup
!ls -lh /content/llama3_resume_minimal.tar.gz
```

For Llama3, `flat_parameters.pth` is the file actually loaded by `--flatquant-raw-path`. `metadata.json` is for checking the config. The 16GB `pytorch_model-*.bin` files are not required if the base model can be loaded again from Hugging Face.

## Quick Checks

Check GPU:

```python
!nvidia-smi
```

Watch RTN logs:

```python
!tail -n 80 /content/smart-flip/logs/rtn_flip_b3_ppl_first/*.log
```

Watch Llama3 logs:

```python
!tail -n 80 /content/llama3/smart-flip/logs/llama3_flatquant_b3_tune_egbc/*.log
```

List result JSONs:

```python
!ls -lh /content/smart-flip/results/eval | tail
!ls -lh /content/llama3/smart-flip/results/eval | tail
```

CASE RIEENG:

Đúng case này thì **đừng set `RUN_RAW=0`**. Dùng:

```bash
RUN_RAW=auto
SKIP_EXISTING_JSON=1
```

Khi đó script xử lý từng model riêng:

```text
Mistral raw JSON đã có  -> skip raw Mistral
Model khác chưa có raw -> chạy raw
Flip JSON nào đã có    -> skip
Flip JSON nào chưa có  -> chạy tiếp
```

Lệnh 3-bit:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b3_ppl_first \
LOG_DIR=./logs/rtn_flip_b3_ppl_first \
HF_DATASETS_CACHE=./data/cache/hf_datasets/datasets-rtn-flip-b3-ppl-first \
GPU=0 BITS=3 STAGE=ppl RUN_RAW=auto C4_SAMPLES=128 SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b3_ppl_first.sh'
```

Lưu ý quan trọng: Flip **không dùng raw JSON để apply flip**. Raw JSON chỉ dùng để biết raw đã chạy rồi và skip. Mỗi Flip run vẫn load base model rồi apply RTN+Flip theo config `k/f`.

Script sẽ skip Mistral raw nếu có file đúng tên:

```text
/content/smart-flip/results/eval/rtn_raw_Mistral-7B-v0.3_b3_ppl.json
```

Check nhanh:

```python
!ls -lh /content/smart-flip/results/eval/rtn_raw_Mistral-7B-v0.3_b3_ppl.json
```

Nếu bạn chỉ có file cũ:

```text
rtn_raw_Mistral-7B-v0.3_b3.json
```

thì script PPL-first **không coi là đã xong**. Và nếu file cũ chạy trước khi sửa `a_bits=16`, tốt nhất không reuse, nên cho chạy lại raw Mistral đúng setting mới.

Đã sửa hai file:

- [scripts/bash/rtn_flip_b3.sh](/home/hungchan/lab_reflourished/smart-flip/scripts/bash/rtn_flip_b3.sh)
- [scripts/bash/rtn_flip_b4.sh](/home/hungchan/lab_reflourished/smart-flip/scripts/bash/rtn_flip_b4.sh)

Giờ hai script này là bản **final full eval**, không tune nữa:

```text
1. Chạy naive RTN
2. Eval WikiText-2 PPL
3. Eval C4 PPL
4. Eval 5 lm_eval tasks: arc_challenge, arc_easy, boolq, piqa, rte
5. Chạy RTN + Flip cố định k=0.0, f=0.05
6. Eval lại đủ 2 PPL + 5 tasks
```

Tôi cũng bỏ nhánh `RUN_FLIP_TUNE`, `KNEE_VALUES`, `MAX_FLIP_VALUES` khỏi hai script final này. `--include-c4` và `--include-lm-eval` được pass explicit.

Lệnh chạy 3-bit:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b3_v1 \
GPU=0 RUN_RAW=auto SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b3.sh'
```

Lệnh chạy 4-bit:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b4 \
GPU=0 RUN_RAW=auto SKIP_EXISTING_JSON=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b4.sh'
```

Chạy 1 model thì thêm `MODEL_PATH=...`, ví dụ:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B
```

Đã check `bash -n` OK.

## Rerun Llama3.1 RTN And Keep Minimal Continue State

Use this when you want to rerun the suspicious Llama3.1 case but only keep what is needed to continue later. For RTN, the minimal continue state is:

```text
results/eval/*.json
logs/*
```

The saved model weights under `results/models/...` are not needed for continue. Keep `CLEAN_MODEL_ARTIFACTS=1` so the script deletes heavy files after each completed run.

3-bit Llama3.1 only:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b3_llama31_debug \
GPU=0 RUN_RAW=1 RUN_FLIP=1 SKIP_EXISTING_JSON=0 \
CLEAN_MODEL_ARTIFACTS=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b3.sh'
```

4-bit Llama3.1 only:

```python
!bash -lc 'cd /content/smart-flip && source .venv/bin/activate && \
export HF_HUB_DISABLE_XET=1 && unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && \
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B \
PYTHON_BIN=/content/smart-flip/.venv/bin/python \
WANDB_PROJECT=rtn_flip_b4_llama31_debug \
GPU=0 RUN_RAW=1 RUN_FLIP=1 SKIP_EXISTING_JSON=0 \
CLEAN_MODEL_ARTIFACTS=1 \
stdbuf -oL -eL bash scripts/bash/rtn_flip_b4.sh'
```

Backup minimal continue state to Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!bash -lc 'set -euo pipefail; \
BACKUP_ROOT="/content/drive/MyDrive/smartflip_rtn_llama31_minimal_$(date +%Y%m%d_%H%M%S)"; \
mkdir -p "$BACKUP_ROOT/results" "$BACKUP_ROOT/logs"; \
cp -r /content/smart-flip/results/eval "$BACKUP_ROOT/results/eval"; \
[ -d /content/smart-flip/logs/rtn_flip_b3 ] && cp -r /content/smart-flip/logs/rtn_flip_b3 "$BACKUP_ROOT/logs/"; \
[ -d /content/smart-flip/logs/rtn_flip_b4 ] && cp -r /content/smart-flip/logs/rtn_flip_b4 "$BACKUP_ROOT/logs/"; \
du -sh "$BACKUP_ROOT"; \
echo "DONE: $BACKUP_ROOT"'
```

Restore minimal continue state in a new session:

```python
BACKUP_ROOT = "/content/drive/MyDrive/smartflip_rtn_llama31_minimal_YYYYMMDD_HHMMSS"
```

```python
!bash -lc 'set -euo pipefail; \
BACKUP_ROOT="'"$BACKUP_ROOT"'"; \
mkdir -p /content/smart-flip/results /content/smart-flip/logs; \
rm -rf /content/smart-flip/results/eval; \
cp -r "$BACKUP_ROOT/results/eval" /content/smart-flip/results/eval; \
[ -d "$BACKUP_ROOT/logs/rtn_flip_b3" ] && cp -r "$BACKUP_ROOT/logs/rtn_flip_b3" /content/smart-flip/logs/; \
[ -d "$BACKUP_ROOT/logs/rtn_flip_b4" ] && cp -r "$BACKUP_ROOT/logs/rtn_flip_b4" /content/smart-flip/logs/; \
echo "Restored minimal RTN continue state."'
```
