"""Language reference data and descriptive distance summaries."""
import json
from pathlib import Path

import numpy as np

from .evaluation import distances

REFERENCE = Path(__file__).resolve().parents[1] / "data/references/language.json"
CONFIG = json.loads(REFERENCE.read_text())
FAMILIES = CONFIG["families"]
CLOCK = CONFIG["clock_sources"]
LINEAGE = CONFIG["lineage_paths"]
AGES = CONFIG["age_sets"]
CONFIGS = [f"{p}_L{L:02d}" for p in ("mean_pool", "high_freq") for L in (4, 8, 12, 18)]


def short(code):
    return code.split("_")[0]


def evaluation_indices(z):
    members = {c for family in FAMILIES.values() for c in family}
    return [i for i, c in enumerate(z["codes"]) if short(str(c)) in members]


def family_distances(z, tag, family, configs):
    """Average per-configuration, mean-normalized Euclidean distances."""
    members = [str(c) for c in z[f"{tag}_members_{family}"]]
    acc = np.zeros((len(members), len(members)))
    for config in configs:
        centroids = z[f"{tag}_{family}_{config}"].astype(np.float64)
        D = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        mean = D[~np.eye(len(members), dtype=bool)].mean()
        acc += D / max(mean, 1e-12)
    return members, acc / len(configs)


def features(codes):
    import lang2vec.lang2vec as l2v
    return {name: l2v.get_features(sorted(codes), name)
            for name in ("fam", "syntax_knn", "syntax_wals", "syntax_average")}


def feature_matrix(mapping, codes):
    def numeric(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan
    return np.array([[numeric(v) for v in mapping[c]] for c in codes], float)


def dense_distance(M, i, j):
    return float(np.linalg.norm(np.nan_to_num(M[i]) - np.nan_to_num(M[j])))


def observed_distance(M, i, j, min_overlap=8):
    mask = ~(np.isnan(M[i]) | np.isnan(M[j]))
    return float(np.abs(M[i, mask] - M[j, mask]).mean()) if mask.sum() >= min_overlap else np.nan


def mrca_years(family, a, b, ages):
    node = {"Romance": "PRom", "Slavic": "PSlav", "Germanic": "PGerm", "Turkic": "PTurk"}[family]
    for x, y in zip(LINEAGE[a], LINEAGE[b]):
        if x != y:
            break
        node = x
    return ages[node]


def cross_script(z):
    codes = [str(c) for c in z["codes"]]
    idx = {short(c): i for i, c in enumerate(codes)}
    pairs = [("hin", "urd"), ("mar", "urd"), ("kaz", "tur"), ("rus", "pol"),
             ("arb", "heb"), ("ben", "hin"), ("tam", "tel"), ("tam", "mal"),
             ("amh", "arb"), ("spa", "por")]
    surface = z["surface_D"]
    iu = np.triu_indices(len(codes), 1)
    sv = surface[iu]
    out = {}
    for metric in ("euclid", "cosine"):
        D = distances(z["glob_mean_pool_L12"], metric)
        values = D[iu]
        for a, b in pairs:
            i, j = idx[a], idx[b]
            row = out.setdefault(f"{a}-{b}", {})
            row[f"geo_{metric}_pct"] = round(float((values < D[i, j]).mean() * 100), 1)
            row["surf_pct"] = round(float((sv < surface[i, j]).mean() * 100), 1)
            row["surf_d"] = round(float(surface[i, j]), 6)
            row["surf_saturated"] = bool(surface[i, j] > .999)
    saturation = dict(median=round(float(np.median(sv)), 6),
        frac_gt_0_99=round(float((sv > .99).mean()), 4),
        frac_gt_0_999=round(float((sv > .999).mean()), 4), max=round(float(sv.max()), 6),
        spread_p60_to_max=round(float(sv.max() - np.percentile(sv, 60)), 6))
    return dict(pairs=out, n_pairs_total=len(sv), surface_saturation=saturation)
