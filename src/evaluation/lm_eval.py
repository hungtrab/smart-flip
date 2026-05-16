"""
lm-evaluation-harness integration for downstream benchmark evaluation.
"""

from __future__ import annotations

import copy
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

    def _clear_isolated_hf_datasets_cache(self):
        cache_dir = os.getenv("HF_DATASETS_CACHE")
        if not cache_dir:
            return False
        path = Path(cache_dir)
        normalized_parts = {part.lower() for part in path.parts}
        if not (path.name.startswith("datasets-") and "hf_datasets" in normalized_parts):
            return False
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        print(f"\nDetected incomplete HF datasets cache, cleared {path} and retrying lm-eval once...")
        return True

    @staticmethod
    def _is_incompatible_hf_cache_type_error(exc: TypeError) -> bool:
        return "must be called with a dataclass type or instance" in str(exc)

    def _augment_incompatible_hf_cache_error(self, exc: TypeError) -> RuntimeError | TypeError:
        if not self._is_incompatible_hf_cache_type_error(exc):
            return exc
        return RuntimeError(
            "lm-eval failed because the Hugging Face datasets cache is incompatible with "
            "the current `datasets` package. This is a dataset-cache problem, not a "
            "quantization/model problem. Clear the isolated HF datasets cache and rerun."
        )

    @staticmethod
    def _is_missing_hf_cache_error(exc: ValueError) -> bool:
        message = str(exc)
        return "Couldn't find cache for" in message and "Available configs in the cache" in message

    def _augment_missing_hf_cache_error(self, exc: ValueError) -> RuntimeError | ValueError:
        if not self._is_missing_hf_cache_error(exc):
            return exc
        return RuntimeError(
            "lm-eval failed because the Hugging Face datasets cache is incomplete. "
            f"Original error: {exc}. "
            "Clear the isolated HF datasets cache and rerun, or prefetch both ARC configs "
            "`ARC-Challenge` and `ARC-Easy`. Also make sure `HF_DATASETS_OFFLINE` is not `1`."
        )

    @staticmethod
    def _is_unreachable_hub_error(exc: Exception) -> bool:
        message = str(exc)
        return "Couldn't reach" in message and "on the Hub" in message

    @staticmethod
    def _is_dataset_probe_unicode_error(exc: Exception) -> bool:
        return isinstance(exc, UnicodeDecodeError)

    def _is_recoverable_dataset_error(self, exc: Exception) -> bool:
        return (
            isinstance(exc, ValueError) and self._is_missing_hf_cache_error(exc)
        ) or self._is_unreachable_hub_error(exc) or self._is_dataset_probe_unicode_error(exc)

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

    def _run_one_task(self, evaluator, model_name: str, model_path, task: str):
        original_tasks = self.tasks
        try:
            self.tasks = [task]
            return self._run_simple_evaluate(evaluator, model_name, model_path)
        finally:
            self.tasks = original_tasks

    def _run_taskwise_best_effort(self, evaluator, model_name: str, model_path):
        merged_payload = None
        skipped = {}

        for task in self.tasks:
            try:
                task_payload = self._run_one_task(evaluator, model_name, model_path, task)
            except Exception as exc:
                if self._is_recoverable_dataset_error(exc):
                    skipped[task] = str(exc)
                    print(f"\nSkipping lm-eval task {task}: {exc}")
                    continue
                raise

            if merged_payload is None:
                merged_payload = task_payload
            else:
                for key in ("results", "configs", "versions", "n-shot"):
                    if isinstance(task_payload.get(key), dict):
                        merged_payload.setdefault(key, {}).update(task_payload[key])
                if isinstance(task_payload.get("samples"), dict):
                    merged_payload.setdefault("samples", {}).update(task_payload["samples"])

        if merged_payload is None:
            raise RuntimeError(
                "All lm-eval tasks failed because the Hugging Face Hub is unreachable "
                "and the required datasets are not fully cached. "
                f"Skipped tasks: {skipped}"
            )

        merged_payload = copy.deepcopy(merged_payload)
        merged_payload["skipped_tasks"] = skipped
        return merged_payload

    def evaluate_model(self, model_name: str, model_path) -> dict:
        try:
            from lm_eval import evaluator
        except ImportError as exc:
            raise RuntimeError(
                "lm-eval is not installed. Install the 'lm-eval' package or disable lm-eval with --no-lm-eval."
            ) from exc

        try:
            payload = self._run_simple_evaluate(evaluator, model_name, model_path)
        except ValueError as exc:
            if self._is_dataset_probe_unicode_error(exc):
                if self._clear_isolated_hf_datasets_cache():
                    try:
                        payload = self._run_simple_evaluate(evaluator, model_name, model_path)
                    except ValueError as retry_exc:
                        if self._is_dataset_probe_unicode_error(retry_exc):
                            payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
                        else:
                            raise self._augment_missing_hf_cache_error(retry_exc) from retry_exc
                    except TypeError as retry_exc:
                        raise self._augment_incompatible_hf_cache_error(retry_exc) from retry_exc
                else:
                    payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
            if self._is_missing_hf_cache_error(exc) and self._clear_isolated_hf_datasets_cache():
                try:
                    payload = self._run_simple_evaluate(evaluator, model_name, model_path)
                except ValueError as retry_exc:
                    if self._is_dataset_probe_unicode_error(retry_exc):
                        payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
                    elif self._is_missing_hf_cache_error(retry_exc):
                        payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
                    else:
                        raise self._augment_missing_hf_cache_error(retry_exc) from retry_exc
                except TypeError as retry_exc:
                    raise self._augment_incompatible_hf_cache_error(retry_exc) from retry_exc
                except UnicodeDecodeError:
                    payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
            elif not self._is_dataset_probe_unicode_error(exc):
                raise self._augment_missing_hf_cache_error(exc) from exc
        except TypeError as exc:
            if self._is_incompatible_hf_cache_type_error(exc) and self._clear_isolated_hf_datasets_cache():
                try:
                    payload = self._run_simple_evaluate(evaluator, model_name, model_path)
                except TypeError as retry_exc:
                    raise self._augment_incompatible_hf_cache_error(retry_exc) from retry_exc
                except ValueError as retry_exc:
                    raise self._augment_missing_hf_cache_error(retry_exc) from retry_exc
                except UnicodeDecodeError:
                    payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
            else:
                raise self._augment_incompatible_hf_cache_error(exc) from exc
        except UnicodeDecodeError as exc:
            if self._clear_isolated_hf_datasets_cache():
                try:
                    payload = self._run_simple_evaluate(evaluator, model_name, model_path)
                except UnicodeDecodeError:
                    payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
            else:
                payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
        except Exception as exc:
            if self._is_recoverable_dataset_error(exc):
                payload = self._run_taskwise_best_effort(evaluator, model_name, model_path)
            else:
                raise
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
