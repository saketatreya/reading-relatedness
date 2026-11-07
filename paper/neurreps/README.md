# Manuscript

**Reading Relatedness and Hierarchy from Frozen Representations**
Saket Atreya and Paras Chopra. NeurReps 2026, Proceedings of Machine Learning Research.

Read [paper.pdf](paper.pdf). The main text ends early on page 9, followed by the references
and appendices.

From the project root, run `make paper`, or run `python3 build.py` from this directory.
The builder regenerates every numeric macro, table and figure from
`../../results/analysis.json`, runs LaTeX and BibTeX, checks for unresolved references
and overfull boxes, and writes `paper.pdf`. Compiler intermediates stay in the ignored
`build/` directory.

`make_figs.py` checks that every `\Result{...}` key used in `paper.tex` exists in the
result record. `generated/numbers.tex` and `generated/number_provenance.json` carry the
result-record hash. The figures are `figs/fig1_recovery.pdf`, `figs/fig2_readouts.pdf`
and `figs/fig3_perturbations.pdf`. `jmlr.cls` and `jmlrutils.sty` are the PMLR / NeurReps
2026 class files.

The manuscript can be regenerated from the final JSON without FastME or the input caches.
