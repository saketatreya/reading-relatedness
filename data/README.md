# Inputs for the paper

`representations/` contains the six frozen NPZ inputs used by the analysis:
language, speech, protein, vision, single-cell and additional language encoders.
These compact caches are part of this study; no model download or GPU is needed
to reproduce the reported analyses. Observation-level clouds are not present.

`references/` contains language families, clock populations, lineage paths and
date sets, the protein gene/species lists, the vision metaclass mapping, and
the date-anchor source table. Reference hierarchies for the primary populations
also travel with their cached vectors.

`supplementary/` contains only the two cloud-statistic inputs used in
the appendix: a rounded displacement-magnitude summary and the covariance pair
CSV. Their scope and limitations are stated in the paper.

`provenance/manifest.json` records hashes and original locations in the authors'
development repository. `extraction_sources.zip` preserves
the eight original scripts/configuration files that created these caches. This
small provenance archive is not imported by the analysis and is not another
active pipeline. Its sources used changing upstream model/data dependencies;
fresh extraction is not claimed to reproduce the caches bit-for-bit.

Input and active-code hashes are recorded again in `results/analysis.json` on
each successful analysis run. Run `make verify` to check them.
