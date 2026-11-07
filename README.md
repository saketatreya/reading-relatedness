# Reading Relatedness and Hierarchy from Frozen Representations

Code, data and results for the paper by **Saket Atreya** and **Paras Chopra**
(NeurReps 2026, Proceedings of Machine Learning Research).
**[Read the paper (PDF)](paper/neurreps/paper.pdf)**

What does agreement between representation geometry and a reference hierarchy establish?
The study separates pairwise relatedness, local quartet agreement and recovery of reference
splits in a reconstructed tree. It evaluates all three across frozen text, speech, protein, vision and single-cell
representations, with matched readout controls, exact-distance validation and a synthetic
perturbation experiment.

## Layout

```text
experiments/       the CPU analysis package
  analyze.py       complete analysis entry point
  evaluation.py    shared scorer and reconstruction adapters (NJ, FastME)
  language.py      language analysis helpers
  synthetic.py     Brownian tree features
  setup_fastme.py  local, pinned FastME build
  verify.py        provenance checks
data/
  representations/ six frozen NPZ representation caches
  references/      populations, reference hierarchies, dates and metadata
  supplementary/   the two cloud-statistic inputs used in the appendix
  provenance/      original extraction sources and input manifest
results/
  analysis.json    single authoritative result record
tests/             evaluator correctness and invariance checks
paper/neurreps/    manuscript source, generated tables and figures, final PDF
Makefile           setup, test, analyze, paper, reproduce, verify, clean
requirements.txt   pinned Python dependencies
```

## Run

Use Python 3.12 or later (the recorded environment uses 3.14). Building FastME needs a C
compiler, make, autoconf and automake. Building the paper needs `pdflatex` and `bibtex`.

```bash
python3 -m pip install -r requirements.txt
make setup         # builds FastME 2.1.6.3 locally, verifying its source checksum
make test          # 13 evaluator checks
make reproduce     # recompute results, build the paper, verify provenance
```

The six compact representation caches are included under `data/representations`, so
no GPU, model extraction or model download is required. See [data/README.md](data/README.md)
for their scope and original extraction sources. FastME is the only downloaded build
dependency after the Python packages are installed.

The individual stages:

```bash
make analyze       # data -> results/analysis.json
make paper         # results -> tables, figures, PDF (no FastME or input caches needed)
make verify        # check input and source hashes and generated result keys
make clean         # remove compiler and Python intermediates
```

Analysis runs use one BLAS thread and 2,000 source-label permutations. Results are
replaced only after the entire run succeeds; interrupted-run progress stays in the
ignored `experiments/.cache/`. Every number, table and figure in the paper is generated
from `results/analysis.json`.

The study does not infer absent phylogenetic information from a failed distance readout.
Nor does it establish a real-world causal mechanism; the paper states the scope and limits.

## Citation

```bibtex
@InProceedings{atreya26,
  title     = {Reading Relatedness and Hierarchy from Frozen Representations},
  author    = {Atreya, Saket and Chopra, Paras},
  booktitle = {Symmetry and Geometry in Neural Representations},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  note      = {To appear}
}
```
