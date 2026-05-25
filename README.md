# smart-flip

`smart-flip` is a research and experimentation repo for quantization, centered on:

- `awq`
- `flatquant`
- the `smart_flip` and `bias_correction` post-correction stages
- perplexity evaluation and the `lm-evaluation-harness`

The main entrypoint is `main.py`. The scripts under `scripts/bash/` are just wrappers for running common recipes quickly.

> **Supported quantization methods:** only `awq` and `flatquant`.
> The `gptq` backend has been fully removed from the repo (quantizer, `--gptq-*`
> CLI flags, wrapper scripts and related tests). FlatQuant still performs
> round-to-nearest weight quantization through the `rtn_utils.py` helper. Any
> leftover `gptq` strings live only inside the vendored FlatQuant library
> (`flatquant/`) and in `legacy/`, neither of which is reached by the main flow.

## Repository layout

- `main.py`: main CLI for quantization and evaluation
- `src/quantization/`: quantization pipeline, AWQ, FlatQuant adapter, bias correction
- `rtn_utils.py`: round-to-nearest helper (`rtn_fwrd`, `find_qlayers`) used by the FlatQuant flow
- `src/post_correction/`: `smart_flip` and the other correction stages
- `src/evaluation/`: standard evaluation and FlatQuant-specific evaluation
- `flatquant/`: the original FlatQuant code, vendored and reused by this repo
- `scripts/bash/`: `.sh` wrappers to run quickly per model family and recipe
- `datasets/`: local datasets some FlatQuant loaders need
- `data/cache/`: runtime calibration/evaluation cache
- `results/models/`: model artifacts after quantization
- `results/eval/`: evaluation result JSON
- `legacy/`: old scripts and docs kept for reference

## Installation

```bash
pip install -r requirements.txt
```

If you use a private Hugging Face model or want to log to W&B, create a `.env` file at the repo root:

```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

`main.py` loads `.env` automatically if the file exists.

## How the repo works

There are two main flows:

1. `float_model`
   Evaluate the original float model.
2. `quantize`
   Quantize and then evaluate right after.

`quantize` is configured by:

- `--origin-method awq|flatquant`
- `--post-correction none|smart_flip|bias_correction`

The remaining modes are essentially shortcuts:

- `raw_quantize` = `post_correction=none`
- `flip_quantize` = `post_correction=smart_flip`
- `compare_all` = evaluate `float`, `raw`, and `flip` together

## Model path resolution

`--model-path` is resolved in this order:

1. use it directly if it is an existing local path
2. try `<models_root>/<model_path>`, where `--models-root` defaults to `/models`
3. otherwise treat it as a Hugging Face model id

Examples:

- `--model-path /models/Mistral-7B-v0.3`
- `--model-path Mistral-7B-v0.3 --models-root /models`
- `--model-path mistralai/Mistral-7B-v0.3`

## Available CLI

See the help:

```bash
python main.py -h
python main.py quantize -h
```

### 1. Evaluate a float model

```bash
python main.py float_model \
  --model-path mistralai/Mistral-7B-v0.3
```

### 2. AWQ raw

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction none \
  --bits 4 \
  --run-name awq_raw_mistral
```

### 3. AWQ + smart_flip

Standard run for AWQ on Mistral with `max_flip_percent = 0.05` and `knee_tolerance = 0`:

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction smart_flip \
  --bits 4 \
  --knee-tolerance 0 \
  --max-flip-percent 0.05 \
  --run-name awq_smart_flip_mistral
```

### 4. AWQ + bias_correction

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method awq \
  --post-correction bias_correction \
  --bits 4 \
  --bias-correction-samples 4096 \
  --run-name awq_bias_correction_mistral
```

### 5. FlatQuant raw

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction none \
  --bits 4 \
  --flatquant-epochs 15 \
  --flatquant-cali-bsz 4 \
  --flatquant-lr 5e-3 \
  --run-name flatquant_raw_mistral
```

### 6. FlatQuant + smart_flip

With `flatquant`, correction recipes reference the previously produced raw artifact via `--flatquant-raw-path`.

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction smart_flip \
  --bits 4 \
  --knee-tolerance 0.02 \
  --max-flip-percent 0.03 \
  --flatquant-raw-path ./results/models/flatquant_raw/flatquant_raw_mistral \
  --run-name flatquant_smart_flip_mistral
```

### 7. FlatQuant + bias_correction

```bash
python main.py quantize \
  --model-path mistralai/Mistral-7B-v0.3 \
  --origin-method flatquant \
  --post-correction bias_correction \
  --bits 4 \
  --bias-correction-samples 4096 \
  --flatquant-raw-path ./results/models/flatquant_raw/flatquant_raw_mistral \
  --run-name flatquant_bias_correction_mistral
```

### 8. Compare all

```bash
python main.py compare_all \
  --model-path mistralai/Mistral-7B-v0.3 \
  --raw-path ./results/models/awq_raw/awq_raw_mistral \
  --flip-path ./results/models/awq_smart_flip/awq_smart_flip_mistral
```

## Evaluation

Every evaluation run writes JSON into `results/eval/`.

The repo has two evaluation flows:

- the default flow in `src/evaluation/sliding_window.py`
  - downloads WikiText-2 and C4 via Hugging Face
  - caches into `data/cache/eval`
- the FlatQuant flow in `src/evaluation/flatquant_runner.py`
  - uses the local dataset scripts under `datasets/`
  - requires local data under `smart-flip/datasets`

Defaults:

- WikiText-2 enabled
- C4 enabled
- `lm_eval` enabled
- default `lm_eval` preset is `extended`

Some useful options:

- `--no-c4`
- `--no-lm-eval`
- `--lm-eval-task-preset core|extended`
- `--lm-eval-tasks ...`
- `--use-wandb`

## Datasets and cache

Two distinct concepts:

1. `data/cache/...`
   Runtime cache the repo creates while running calibration/evaluation.
2. `datasets/...`
   Local datasets that some FlatQuant loaders read directly.

### Default datasets for `main.py`

For the usual AWQ flows and sliding-window evaluation, the repo downloads data via Hugging Face:

- `c4` calibration in `src/calibration.py`
- WikiText-2 test in `src/evaluation/sliding_window.py`
- C4 validation in `src/evaluation/sliding_window.py`

You do not need to manually download anything into `smart-flip/datasets` just to use these flows.

### Local datasets required by FlatQuant

The FlatQuant loaders in the following files read local datasets:

- `flatquant/data_utils.py`
- `src/evaluation/flatquant_data_utils.py`

They look for data in:

- `datasets/wikitext`
- `datasets/allenai/c4`
- `datasets/ptb_text_only`
- `datasets/pile-val-backup`

In practice, at minimum prepare `datasets/wikitext`, since the repo ships a dataset script and this is the most common pitfall when running the FlatQuant evaluation flow.

### Downloading WikiText-2 into `smart-flip/datasets`

The repo ships the script `datasets/wikitext/wikitext.py`, which reads a local zip file:

- `datasets/wikitext/wikitext-2-raw-v1.zip`

How to download:

```bash
mkdir -p datasets/wikitext
cd datasets/wikitext
wget -O wikitext-2-raw-v1.zip \
  "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip?download=true"
cd /workspace/smart-flip
```

After downloading, the layout should look like:

```text
smart-flip/
  datasets/
    wikitext/
      wikitext.py
      wikitext-2-raw-v1.zip
```

### Other local datasets for FlatQuant

If you use the corresponding FlatQuant loaders, place the data as follows:

- `datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz`
- `datasets/allenai/c4/en/c4-validation.00000-of-00008.json.gz`
- `datasets/ptb_text_only/...`
- `datasets/pile-val-backup/...`

Notes:

- this README does not ship an auto-download script for `c4`, `ptb_text_only`, `pile-val-backup`
- if you do not call those loaders, you do not need to download them
- `src/calibration.py` and `src/evaluation/sliding_window.py` can already fetch data from Hugging Face

## Quick runs with `.sh` files

The repo ships wrapper scripts grouped by:

- `scripts/bash/smart_flip/awq/`
- `scripts/bash/smart_flip/flatquant/`
- `scripts/bash/bias_correction/awq/`
- `scripts/bash/bias_correction/flatquant/`

Each group has scripts for:

- `run_mistral.sh`
- `run_llama3.sh`
- `run_llama31.sh`
- `run_qwen25.sh`

### Basic usage

For example, with Mistral:

```bash
bash scripts/bash/smart_flip/awq/run_mistral.sh
```

Or override the model:

```bash
MODEL_PATH=meta-llama/Meta-Llama-3-8B \
MODELS_ROOT=/models \
bash scripts/bash/smart_flip/awq/run_llama3.sh
```

### `smart_flip/awq` scripts

The script will:

1. run `float_model`
2. run `awq raw`
3. sweep the `knee_tolerance` x `max_flip_percent` grid for `smart_flip`

Example:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
bash scripts/bash/smart_flip/awq/run_mistral.sh
```

### `bias_correction/awq` scripts

The script will:

1. optionally run `float_model`
2. run `awq raw`
3. run `awq + bias_correction`

Example:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
BIAS_CORRECTION_SAMPLES=4096 \
bash scripts/bash/bias_correction/awq/run_mistral.sh
```

### `smart_flip/flatquant` scripts

The script will:

1. optionally run `float_model`
2. optionally run `flatquant raw`
3. if raw is skipped, reuse `RAW_MODEL_DIR`
4. run `flatquant + smart_flip` with `--flatquant-raw-path "$RAW_MODEL_DIR"`

Example:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
bash scripts/bash/smart_flip/flatquant/run_mistral.sh
```

If you already have a raw artifact:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
RUN_RAW_QUANTIZE=0 \
RAW_MODEL_DIR=./results/models/flatquant_raw/flatquant_raw_Mistral-7B-v0.3 \
bash scripts/bash/smart_flip/flatquant/run_mistral.sh
```

### `bias_correction/flatquant` scripts

Similarly, this script produces a raw FlatQuant artifact first, then runs correction:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
bash scripts/bash/bias_correction/flatquant/run_mistral.sh
```

Or reuse a raw artifact:

```bash
MODEL_PATH=mistralai/Mistral-7B-v0.3 \
RUN_RAW_QUANTIZE=0 \
RAW_MODEL_DIR=./results/models/flatquant_raw/flatquant_raw_Mistral-7B-v0.3 \
bash scripts/bash/bias_correction/flatquant/run_mistral.sh
```

### Common environment variables for `.sh`

All wrapper scripts support these environment variables:

- `MODEL_PATH`
- `MODELS_ROOT`
- `PYTHON_BIN`
- `RESULTS_MODELS_DIR`
- `RESULTS_EVAL_DIR`
- `CALIBRATION_CACHE_DIR`
- `EVAL_CACHE_DIR`
- `CALIB_DATASET`
- `N_CALIB`
- `CALIB_SEQLEN`
- `SEED`
- `STRIDE`
- `MAX_LENGTH`
- `C4_SAMPLES`
- `LM_EVAL_TASK_PRESET`
- `INCLUDE_LM_EVAL`
- `INCLUDE_C4`
- `USE_WANDB`
- `WANDB_PROJECT`
- `WANDB_ENTITY`

FlatQuant wrappers additionally support:

- `RUN_FLOAT_MODEL`
- `RUN_RAW_QUANTIZE`
- `RAW_MODEL_DIR`
- `FLATQUANT_EPOCHS`
- `FLATQUANT_CALI_BSZ`
- `FLATQUANT_LR`
- `FLATQUANT_DIAG_INIT`
- `FLATQUANT_DIAG_ALPHA`
- `FLATQUANT_CALI_TRANS`
- `FLATQUANT_ADD_DIAG`
- `FLATQUANT_LWC`
- `FLATQUANT_LAC`

## Testing

The repo uses `unittest`, run via `pytest`:

```bash
# Full suite
python -m pytest tests/ -v

# A single file
python -m pytest tests/test_quantization_pipeline.py -v

# A single test
python -m pytest tests/test_main.py::ParserModeTests::test_parser_accepts_single_model_modes -v
```

Key test files:

- `tests/test_main.py`: CLI parser and the `run_quantize` flow
- `tests/test_quantization_pipeline.py`: pipeline factory and config (AWQ, FlatQuant, post-correction)
- `tests/test_bash_scripts.py`: layout and content checks for the `.sh` wrappers
- `tests/test_flatquant_*.py`, `tests/test_lm_eval_runner.py`: FlatQuant and lm-eval integration

## Output locations

- model artifacts: `results/models/<variant>/<run_name>/`
- evaluation JSON: `results/eval/<run_name>.json`
- model metadata: `results/models/<variant>/<run_name>/metadata.json`

Notes:

- for `flatquant` + `post_correction=none`, the raw output is retained so correction stages can reuse it
- some temporary quantized outputs may be deleted after evaluation if the pipeline does not need to keep them

## Notes

- `smart_flip` and `bias_correction` are both post-correction stages.
- `flatquant` needs more complete local datasets than `awq`, especially when running through the FlatQuant evaluation/loader flow.
- If you hit dataset errors while running FlatQuant, check `smart-flip/datasets` first.
- The scripts under `legacy/` are kept for reference and are no longer part of the main flow.
