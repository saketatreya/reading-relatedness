"""Check that results and the manuscript use the current code and input data."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / "results/analysis.json"
    result = json.loads(path.read_text())
    if result["schema_version"] != 2:
        raise ValueError("Run make analyze to produce the consolidated result schema.")
    for group in ("inputs", "sources"):
        for name, expected in result["provenance"][group].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Changed {group}: {name}; rerun make reproduce.")
    manuscript = ROOT / "paper/neurreps"
    numbers = json.loads((manuscript / "generated/number_provenance.json").read_text())
    required = set(re.findall(r"\\Result\{([^{}]+)\}", (manuscript / "paper.tex").read_text()))
    if required - numbers.keys():
        raise ValueError(f"Missing manuscript results: {sorted(required - numbers.keys())}")
    if digest(path) not in (manuscript / "generated/numbers.tex").read_text().splitlines()[0]:
        raise ValueError("Manuscript numbers are stale; run make paper.")
    if not (manuscript / "paper.pdf").is_file():
        raise FileNotFoundError("Build the paper with make paper.")
    print(f"Verified {len(result['provenance']['inputs'])} inputs, "
          f"{len(result['provenance']['sources'])} analysis modules, and "
          f"{len(required)} manuscript result keys.")


if __name__ == "__main__":
    main()
