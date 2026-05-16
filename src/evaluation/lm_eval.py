"""
lm-evaluation-harness integration for downstream benchmark evaluation.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.io_utils import dump_json


class LMEvalHarnessRunner:
    def __init__(
        self,
        tasks: list[str],
        device: str = "cuda",
        batch_size: str = "auto",
        num_fewshot: int | None = None,
        output_dir: str = "./results/eval/lm_eval",
        run_name: str | None = None,
        hf_token: str | None = None,
    ):
        self.tasks = tasks
        self.device = device
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot
        self.output_dir = Path(output_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.hf_token = hf_token
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _augment_known_runtime_error(self, exc: RuntimeError) -> RuntimeError:
        message = str(exc)
        if "Dataset scripts are no longer supported" not in message:
            return exc

        dataset_script = None
        if "found " in message:
            dataset_script = message.rsplit("found ", 1)[-1].strip()

        script_hint = f" ({dataset_script})" if dataset_script else ""
        return RuntimeError(
            "lm-eval is running with an incompatible Hugging Face `datasets` version. "
            f"Hugging Face removed dataset loading scripts in `datasets>=4.0`, and one of the requested tasks still needs one{script_hint}. "
            "Install a compatible version such as `datasets==2.17.1` (or any `datasets<4`) and rerun. "
            "If you only need perplexity, rerun with `INCLUDE_LM_EVAL=0` or pass `--no-lm-eval`."
        )

    def _augment_known_unicode_error(self, exc: UnicodeDecodeError) -> RuntimeError:
        message = str(exc)
        if "invalid start byte" not in message:
            return RuntimeError(f"lm-eval failed while loading a Hugging Face dataset: {message}")

        return RuntimeError(
            "lm-eval hit a known Hugging Face `datasets` bug while probing a dataset from the Hub. "
            "This has been observed with newer `datasets` 2.17/2.18 releases on some tasks. "
            "For the isolated Llama env, recreate the env with `requirements_llama3.txt`, which pins "
            "`lm-eval==0.4.1`, `datasets==2.14.6`, and `pyarrow==20.0.0`, then clear the old cache and rerun. "
            "In practice, clear both `~/.cache/huggingface/datasets` and dataset repos under "
            "`~/.cache/huggingface/hub/datasets--*`. "
            "If you only need perplexity, rerun with `INCLUDE_LM_EVAL=0` or pass `--no-lm-eval`."
        )

    def _augment_known_type_error(self, exc: TypeError) -> TypeError:
        message = str(exc)
        if "must be called with a dataclass type or instance" not in message:
            return exc

        return TypeError(
            "lm-eval hit an incompatible Hugging Face datasets cache. "
            "This usually happens when the current environment uses an older `datasets` version, "
            "but the local HF dataset cache was created by a newer version. "
            "Use a version-isolated HF datasets cache or delete the stale task cache and rerun. "
            "This repo now defaults `HF_DATASETS_CACHE` to `./data/cache/hf_datasets/datasets-<version>`; "
            "if you still see this on an existing server, clear the old global cache under `~/.cache/huggingface/datasets` "
            "for the affected tasks, or rerun with `INCLUDE_LM_EVAL=0` if you only need perplexity."
        )

    def _augment_known_import_error(self, exc: ImportError) -> RuntimeError:
        message = str(exc)

        if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None) == "lm_eval":
            return RuntimeError(
                "lm-eval is not installed. Install the `lm-eval` package or disable lm-eval with `--no-lm-eval`."
            )

        if "huggingface_hub.errors" in message:
            return RuntimeError(
                "lm-eval is installed, but one of its dependencies is incompatible: "
                "`peft` is importing `huggingface_hub.errors`, which is missing from the currently installed "
                "`huggingface-hub` version. For the Llama env, install a newer hub release such as "
                "`huggingface-hub==0.24.6` and rerun. "
                "If you only need perplexity, rerun with `INCLUDE_LM_EVAL=0` or pass `--no-lm-eval`."
            )

        return RuntimeError(
            "lm-eval could not be imported because one of its dependencies failed to import. "
            f"Original import error: {message}"
        )

    def _hf_datasets_cache_dir(self) -> Path | None:
        cache_dir = os.getenv("HF_DATASETS_CACHE")
        if not cache_dir:
            return None
        return Path(cache_dir)

    def _can_clear_hf_datasets_cache(self, cache_dir: Path) -> bool:
        normalized_parts = {part.lower() for part in cache_dir.parts}
        return cache_dir.name.startswith("datasets-") and "hf_datasets" in normalized_parts

    def _clear_hf_datasets_cache(self):
        cache_dir = self._hf_datasets_cache_dir()
        if cache_dir is None:
            return False
        if not self._can_clear_hf_datasets_cache(cache_dir):
            return False

        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDetected incompatible HF datasets cache, cleared {cache_dir} and retrying lm-eval once...")
        return True

    @staticmethod
    def _is_missing_hf_cache_error(exc: ValueError) -> bool:
        message = str(exc)
        return "Couldn't find cache for" in message and "Available configs in the cache" in message

    def _augment_known_value_error(self, exc: ValueError) -> RuntimeError | ValueError:
        if not self._is_missing_hf_cache_error(exc):
            return exc

        return RuntimeError(
            "lm-eval failed because the Hugging Face datasets cache is incomplete. "
            f"Original error: {exc}. "
            "This is a dataset-cache problem, not a quantization/model problem. "
            "Clear the isolated HF datasets cache and rerun, or prefetch both ARC configs "
            "`ARC-Challenge` and `ARC-Easy` before launching the experiment. "
            "Also make sure `HF_DATASETS_OFFLINE` is not set to `1`."
        )

    def _run_simple_evaluate(self, evaluator, model_name: str, model_path):
        if isinstance(model_path, dict) and {"model", "tokenizer"}.issubset(model_path):
            from lm_eval.models.huggingface import HFLM

            model = model_path["model"]
            tokenizer = model_path["tokenizer"]
            if self.device == "cuda":
                model = model.to(self.device)
            eval_batch_size = 1 if self.batch_size == "auto" else self.batch_size
            hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)
            return evaluator.simple_evaluate(
                model=hflm,
                tasks=self.tasks,
                device=self.device,
                batch_size=eval_batch_size,
                num_fewshot=self.num_fewshot,
                log_samples=False,
            )

        return evaluator.simple_evaluate(
            model="hf",
            model_args=self._model_args(model_path),
            tasks=self.tasks,
            device=self.device,
            batch_size=self.batch_size,
            num_fewshot=self.num_fewshot,
            log_samples=False,
        )

    def _model_args(self, model_path: str) -> str:
        dtype = "float16" if self.device == "cuda" else "float32"
        model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
        if self.hf_token:
            model_args += f",token={self.hf_token}"
        return model_args

    def _make_json_safe(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if callable(value):
            name = getattr(value, "__name__", value.__class__.__name__)
            return f"<callable {name}>"

        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(item) for item in value]

        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return self._make_json_safe(item_method())
            except (TypeError, ValueError):
                pass

        return repr(value)

    def _summarize_results(self, payload: dict) -> dict:
        results = payload.get("results", {})
        summary = {}
        for task_name, metrics in results.items():
            task_summary = {}
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    task_summary[metric_name] = value
            summary[task_name] = task_summary
        return summary

    def _write_raw_results(self, model_name: str, payload: dict):
        output_path = self.output_dir / f"{self.run_name}_{model_name}.json"
        safe_payload = self._make_json_safe(payload)
        dump_json(output_path, safe_payload, indent=2)

    def evaluate_model(self, model_name: str, model_path) -> dict:
        try:
            from lm_eval import evaluator
        except ImportError as exc:
            raise self._augment_known_import_error(exc) from exc

        try:
            payload = self._run_simple_evaluate(evaluator, model_name, model_path)
        except RuntimeError as exc:
            raise self._augment_known_runtime_error(exc) from exc
        except ValueError as exc:
            if self._is_missing_hf_cache_error(exc) and self._clear_hf_datasets_cache():
                try:
                    payload = self._run_simple_evaluate(evaluator, model_name, model_path)
                except RuntimeError as retry_exc:
                    raise self._augment_known_runtime_error(retry_exc) from retry_exc
                except ValueError as retry_exc:
                    raise self._augment_known_value_error(retry_exc) from retry_exc
                except UnicodeDecodeError as retry_exc:
                    raise self._augment_known_unicode_error(retry_exc) from retry_exc
                except TypeError as retry_exc:
                    raise self._augment_known_type_error(retry_exc) from retry_exc
            else:
                raise self._augment_known_value_error(exc) from exc
        except UnicodeDecodeError as exc:
            raise self._augment_known_unicode_error(exc) from exc
        except TypeError as exc:
            if "must be called with a dataclass type or instance" in str(exc) and self._clear_hf_datasets_cache():
                try:
                    payload = self._run_simple_evaluate(evaluator, model_name, model_path)
                except RuntimeError as retry_exc:
                    raise self._augment_known_runtime_error(retry_exc) from retry_exc
                except UnicodeDecodeError as retry_exc:
                    raise self._augment_known_unicode_error(retry_exc) from retry_exc
                except TypeError as retry_exc:
                    raise self._augment_known_type_error(retry_exc) from retry_exc
            else:
                raise self._augment_known_type_error(exc) from exc
        safe_payload = self._make_json_safe(payload)
        self._write_raw_results(model_name, payload)
        return {
            "tasks": list(self.tasks),
            "summary": self._summarize_results(payload),
            "raw": safe_payload,
        }

    def run(self, model_paths: Dict[str, str]) -> dict:
        results = {}
        for model_name, model_path in model_paths.items():
            print(f"\nRunning lm-eval for {model_name} on {', ' .join(self.tasks)}...")
            results[model_name] = self.evaluate_model(model_name, model_path)
        return results
