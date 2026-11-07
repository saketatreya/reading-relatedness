# One analysis pipeline

Run `make analyze` from the project root. It reads `data/` and atomically writes
`results/analysis.json` after all validity controls and analysis sections pass.

- `analyze.py`: the complete paper analysis and result provenance.
- `evaluation.py`: shared distances, NJ/FastME, splits, quartets and permutations.
- `language.py`: language reference handling and descriptive summaries.
- `synthetic.py`: the Brownian feature generator used in the paper.
- `setup_fastme.py`: checksum-verified local FastME build.
- `verify.py`: input/code hashes and manuscript result provenance.

Use package entry points (`python3 -m experiments.analyze`), not execution of
individual module files. The root README documents dependencies and commands;
the scientific protocol is described in the paper, and tests are in `tests/`.

`.cache/` is ignored local tooling and temporary progress. It is not an input
data store.
