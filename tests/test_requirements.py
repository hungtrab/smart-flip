import re
import unittest
from pathlib import Path


def _package_names(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # strip version specifiers / extras: "lm-eval==0.4.9.1" -> "lm-eval"
        name = re.split(r"[=<>!~\[ ]", stripped, maxsplit=1)[0].strip()
        if name:
            names.add(name)
    return names


class RequirementsTests(unittest.TestCase):
    def test_requirements_txt_exists_and_lists_runtime_dependencies(self):
        path = Path("requirements.txt")
        self.assertTrue(path.exists(), "requirements.txt is missing")

        names = _package_names(path.read_text(encoding="utf-8"))

        # torch is installed separately to match the local CUDA build (see README),
        # so it is intentionally not listed here.
        required = {"numpy", "transformers", "datasets", "tqdm", "lm-eval", "wandb", "python-dotenv"}
        self.assertTrue(required.issubset(names), f"Missing dependencies: {sorted(required - names)}")


if __name__ == "__main__":
    unittest.main()
