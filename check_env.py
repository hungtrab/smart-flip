#!/usr/bin/env python3
"""Quick env check before running tune_egbc_b3.sh"""

import sys
import importlib
import subprocess

OK = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

errors = 0

def check(label, fn):
    global errors
    try:
        result = fn()
        msg = f" ({result})" if result else ""
        print(f"  {OK} {label}{msg}")
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        errors += 1

def check_import(pkg, import_as=None):
    name = import_as or pkg
    check(f"import {name}", lambda: importlib.import_module(name) and None)

def check_version(pkg, import_as=None, attr="__version__"):
    name = import_as or pkg
    def fn():
        m = importlib.import_module(name)
        return getattr(m, attr, "?")
    check(f"import {name}", fn)

# ── Python ────────────────────────────────────────────────────────────────────
print("\n[Python]")
check("version >= 3.10", lambda: f"{sys.version.split()[0]}" if sys.version_info >= (3, 10)
      else (_ for _ in ()).throw(RuntimeError(f"need 3.10+, got {sys.version}")))

# ── Core ML packages ──────────────────────────────────────────────────────────
print("\n[Core packages]")
check_version("torch")
check("torch.cuda.is_available", lambda: f"{torch.cuda.device_count()} GPU(s)"
      if (torch := importlib.import_module("torch")).cuda.is_available()
      else (_ for _ in ()).throw(RuntimeError("no CUDA")))
check_version("transformers")
check_version("datasets")
check_version("accelerate")
check_version("peft")
check_version("numpy")
check_version("tqdm")
check_version("psutil")
check_version("sentencepiece")

# ── lm-eval ───────────────────────────────────────────────────────────────────
print("\n[lm-eval]")
check_version("lm_eval")

# ── W&B ──────────────────────────────────────────────────────────────────────
print("\n[wandb]")
check_version("wandb")
check("WANDB_API_KEY set", lambda: "ok" if __import__("os").environ.get("WANDB_API_KEY")
      else (_ for _ in ()).throw(RuntimeError("not set — wandb logging will fail")))

# ── HuggingFace token ─────────────────────────────────────────────────────────
print("\n[HuggingFace]")
check("HF_TOKEN set", lambda: "ok" if __import__("os").environ.get("HF_TOKEN")
      else (_ for _ in ()).throw(RuntimeError("not set — private models will fail")))
check_import("huggingface_hub")

# ── Project imports ───────────────────────────────────────────────────────────
print("\n[Project modules]")
check_import("src.evaluation.sliding_window")
check_import("src.evaluation.lm_eval")
check_import("src.calibration")
check_import("src.quantization.pipeline")
check_import("src.quantization.awq")
check_import("src.quantization.flatquant")
check_import("src.post_correction.smart_flip")
check_import("flatquant.train_utils")
check_import("flatquant.model_utils")

# ── main.py dry-run ───────────────────────────────────────────────────────────
print("\n[main.py]")
check("python main.py -h", lambda: subprocess.run(
    [sys.executable, "main.py", "-h"],
    capture_output=True, timeout=15
).returncode == 0 and "ok" or (_ for _ in ()).throw(RuntimeError("non-zero exit")))

# ── dotenv ────────────────────────────────────────────────────────────────────
print("\n[.env]")
check_import("dotenv", "dotenv")
check(".env file exists", lambda: "ok" if __import__("pathlib").Path(".env").exists()
      else (_ for _ in ()).throw(RuntimeError(".env not found — HF_TOKEN/WANDB_API_KEY won't auto-load")))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors == 0:
    print(f"{OK} All checks passed — env is ready.")
else:
    print(f"{FAIL} {errors} check(s) failed — fix before running tune_egbc_b3.sh")
    sys.exit(1)
