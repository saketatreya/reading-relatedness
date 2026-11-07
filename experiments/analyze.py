"""Camera-ready audit from immutable cached representations; no GPU or extraction.

Run with OPENBLAS_NUM_THREADS=1 python3 -m experiments.analyze. The output is written only
after all sections succeed. Intermediate progress is kept in ignored .cache/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from scipy.stats import spearmanr

from . import evaluation as ev
from . import language
from .synthetic import gen_latent
from .language import AGES, CLOCK, FAMILIES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FASTME = HERE / ".cache/fastme/bin/fastme"
REPRESENTATIONS = {name: DATA / "representations" / f"{name}.npz"
                   for name in ("language", "speech", "singlecell", "replication", "protein", "vision")}
INPUTS = list(REPRESENTATIONS.values()) + sorted((DATA / "references").glob("*")) + [
    DATA / "supplementary/cloud_magnitude.txt", DATA / "supplementary/cloud_covariance.csv",
    DATA / "provenance/manifest.json", DATA / "provenance/extraction_sources.zip",
]
SOURCES = [HERE / name for name in ("analyze.py", "evaluation.py", "language.py", "synthetic.py")]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def evaluate(X, levels, group_depth, binary, nperm=0, Q=None, extra_depth=None):
    n = len(X)
    G, splits, depths = ev.reference(levels)
    groups = [x[:group_depth] for x in levels]
    if Q is None:
        Q, rule = ev.quartet_bank(n)
    else:
        rule = "shared quartet bank"
    result = dict(n=n, dimension=X.shape[1], n_reference_splits=len(splits),
                  group_depth=group_depth, quartet_rule=rule,
                  levels=[list(x) for x in levels], metrics={})
    for kind in ("euclid", "cosine"):
        D = ev.distances(X, kind)
        rec = ev.direct_scores(D, G, groups, Q)
        if extra_depth is not None:
            extra_groups = [x[:extra_depth] for x in levels]
            rec["nn_finer"] = ev.nn_purity(D, extra_groups)
            rec["nn_finer_chance"] = ev.nn_chance(extra_groups)
        tree = ev.nj(D)
        rec["nj"] = ev.score_tree(tree, splits, G, Q)
        tree, newick = ev.fastme(D, binary)
        rec["fastme"] = ev.score_tree(tree, splits, G, Q)
        rec["fastme"]["newick"] = newick
        if nperm:
            rec["permutation"] = ev.permutation_tests(D, G, groups, Q, nperm)
        result["metrics"][kind] = rec
    return result


def get_datasets():
    lang, speech, cell, repl, protein, vision = [load(p) for p in REPRESENTATIONS.values()]
    sl = [(("IE" if f in {"Romance", "Germanic", "Slavic", "IndoAryan"} else f),
           str(f), str(c)) for f, c in zip(speech["fams"], speech["labels"])]
    cl = [(str(f), str(c)) for f, c in zip(cell["lineages"], cell["labels"])]
    datasets = {
        "text": (lang["glob_mean_pool_L12"], [tuple(x) for x in lang["levels"]], 2, 3),
        "speech": (speech["C_L12"], sl, 2, None),
        "protein": (protein["X"], [tuple(x) for x in protein["levels"]], 1, 2),
        "vision": (vision["X"], [tuple(x) for x in vision["levels"]], 1, 2),
        "singlecell": (cell["C"], cl, 1, None),
    }
    return datasets, lang, speech, repl


def exact_controls(binary):
    out = {}
    for depth in (3, 8):
        n = 2 ** depth
        levels = [tuple((i >> k) & 1 for k in reversed(range(depth))) for i in range(n)]
        G, splits, _ = ev.reference(levels)
        Q, _ = ev.quartet_bank(n)
        row = dict(n=n, n_reference_splits=len(splits))
        for name in ("nj", "fastme"):
            tree = ev.nj(G) if name == "nj" else ev.fastme(G, binary)[0]
            score = ev.score_tree(tree, splits, G, Q)
            assert score["split_recall"] == 1 and score["tree_quartet"] == 1
            assert score["full_tree_exact"]
            row[name] = score
        groups = [p[:-1] for p in levels]
        row["direct"] = ev.direct_scores(G, G, groups, Q)
        out[str(n)] = row
    return out


def domain_results(datasets, binary, nperm):
    out = {}
    for name, (X, levels, depth, finer) in datasets.items():
        print(f"  domain {name}: n={len(X)}, permutations={nperm}", flush=True)
        out[name] = evaluate(X, levels, depth, binary, nperm, extra_depth=finer)
    pc = json.loads((DATA / "references/protein.json").read_text())
    X = datasets["protein"][0]
    assert X.shape[1] % len(pc["GENES"]) == 0
    blocks = X.reshape(len(X), len(pc["GENES"]), -1)
    out["protein"]["genes"] = pc["GENES"]
    out["protein"]["zero_imputed_blocks_per_species"] = [
        int(np.all(b == 0, axis=1).sum()) for b in blocks]
    out["vision"]["metaclass_mapping"] = json.loads((DATA / "references/vision.json").read_text())["META"]
    return out


def feature_distances(codes):
    langs = [language.short(str(c)) for c in codes]
    f = language.features(sorted(set(langs)))
    mats = {k: language.feature_matrix(v, langs) for k, v in f.items()}
    n = len(langs)
    out = {}
    for name, M in mats.items():
        D = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            value = (language.observed_distance(M, i, j)
                     if name in ("syntax_average", "syntax_wals")
                     else language.dense_distance(M, i, j))
            D[i, j] = D[j, i] = value
        out[name] = D
    return out


def adjusted_readout(X, G, T, frag, center=False):
    X = np.asarray(X, float)
    current = X - X.mean(0) if center else X
    iu = np.triu_indices(len(X), 1)
    out = dict(norm_fragmentation_r=ev.corr(np.linalg.norm(current, axis=1), frag))
    for metric in ("euclid", "cosine"):
        D = ev.distances(current, metric)
        r = ev.partial(D[iu], G[iu], [T[iu]])
        deleted = []
        for i in range(len(X)):
            keep = np.arange(len(X)) != i
            Y = X[keep]
            if center:
                Y = Y - Y.mean(0)
            ij = np.triu_indices(len(Y), 1)
            dy = ev.distances(Y, metric)
            deleted.append(ev.partial(dy[ij], G[np.ix_(keep, keep)][ij],
                                      [T[np.ix_(keep, keep)][ij]]))
        out[metric] = dict(r=r, source_deletion_min=min(deleted),
                           source_deletion_max=max(deleted), source_deletion_values=deleted)
    return out


def readout_results(lang, speech, datasets, domains, binary):
    out = {"language": {}, "speech": {}}
    idx = language.evaluation_indices(lang)
    for population, selected in (("36", idx), ("89", list(range(len(lang["codes"]))))):
        codes = [str(lang["codes"][i]) for i in selected]
        feat = feature_distances(codes)
        frag = lang["frag"][selected]
        frame = dict(n=len(selected), codes=codes, layers={})
        for L in (4, 8, 12, 18):
            cfg = f"mean_pool_L{L:02d}"
            raw = lang[f"raw_{cfg}"][selected]
            glob = lang[f"glob_{cfg}"][selected]
            variants = {"raw": (raw, False), "raw_centered": (raw, True),
                        "aligned89": (glob, False), "aligned89_centered": (glob, True)}
            if population == "36":
                variants.update(aligned36=(lang[f"glob36_{cfg}"], False),
                                aligned36_centered=(lang[f"glob36_{cfg}"], True))
            frame["layers"][str(L)] = {}
            for name, (X, center) in variants.items():
                frame["layers"][str(L)][name] = adjusted_readout(
                    X, feat["fam"], feat["syntax_knn"], frag, center)
        out["language"][population] = frame
    _, sl, depth, _ = datasets["speech"]
    for L in (6, 12, 18, 24):
        X = np.asarray(speech[f"C_L{L:02d}"], float)
        if L == 12:
            row = json.loads(json.dumps(domains["speech"]))
        else:
            row = evaluate(X, sl, depth, binary)
        Y = X / np.linalg.norm(X, axis=1, keepdims=True)
        sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
        row["participation_ratio"] = float(np.sum(sv ** 2) ** 2 / np.sum(sv ** 4))
        row["max_centered_rank"] = min(X.shape[0] - 1, X.shape[1])
        row["mean_pairwise_cosine"] = float((Y @ Y.T)[np.triu_indices(len(Y), 1)].mean())
        out["speech"][str(L)] = row
    return out


def family_pairs(z, configs):
    fams = list(FAMILIES)
    codes = sorted({language.short(str(c)) for f in fams for c in z[f"famA_members_{f}"]})
    features = feature_distances(codes)
    lookup = {c: i for i, c in enumerate(codes)}
    rows, pair_indices = [], []
    for fam in fams:
        members, D = language.family_distances(z, "famA", fam, configs)
        for a, b in combinations(range(len(members)), 2):
            i, j = lookup[language.short(members[a])], lookup[language.short(members[b])]
            pair_indices.append((i, j))
            rows.append([D[a, b], features["fam"][i, j],
                         features["syntax_knn"][i, j],
                         features["syntax_average"][i, j],
                         features["syntax_wals"][i, j]])
    A = np.asarray(rows)
    pairs = np.asarray(pair_indices)

    def estimate(mask, ctrl, reverse=False):
        r = ev.partial(A[mask, 0], A[mask, ctrl if reverse else 1],
                       [A[mask, 1 if reverse else ctrl]])
        values = []
        for i in range(len(codes)):
            keep = mask & np.all(pairs != i, axis=1)
            values.append(ev.partial(A[keep, 0], A[keep, ctrl if reverse else 1],
                                     [A[keep, 1 if reverse else ctrl]]))
        return dict(r=r, n_pairs=int(mask.sum()), source_deletion_min=min(values),
                    source_deletion_max=max(values), source_deletion_values=values)

    full = np.ones(len(A), bool)
    matched = np.isfinite(A[:, 3])
    wals = np.isfinite(A[:, 4])
    out = dict(n_sources=len(codes), n_pairs=len(A), codes=codes,
               unadjusted_r=ev.corr(A[:, 0], A[:, 1]),
               imputed_all=estimate(full, 2),
               imputed_matched=estimate(matched, 2),
               observed_matched=estimate(matched, 3),
               wals=estimate(wals, 4),
               reverse_observed=estimate(matched, 3, reverse=True))
    return out


def clock_block(z, configs, ages, nperm, seed=0):
    per_family, blocks = {}, []
    for family, codes in CLOCK.items():
        members, D = language.family_distances(z, "famA", family, configs)
        lookup = {language.short(c): i for i, c in enumerate(members)}
        idx = [lookup[c] for c in codes]
        geom = D[np.ix_(idx, idx)]
        T = np.zeros_like(geom)
        for i, j in combinations(range(len(codes)), 2):
            T[i, j] = T[j, i] = language.mrca_years(family, codes[i], codes[j], ages)
        iu = np.triu_indices(len(T), 1)
        g, t = geom[iu], T[iu]
        per_family[family] = dict(n_sources=len(codes), n_pairs=len(g),
                                  r=ev.corr(g, t), rho=float(spearmanr(g, t).statistic))
        blocks.append((family, codes, geom, T))

    def vectors(selected, permutations=None, drop=None):
        gg, tt = [], []
        for family, codes, D, T in selected:
            keep = [i for i, c in enumerate(codes) if (family, c) != drop]
            gmat = D[np.ix_(keep, keep)]
            tmat = T[np.ix_(keep, keep)]
            if permutations is not None:
                p = permutations[family]
                tmat = tmat[np.ix_(p, p)]
            iu = np.triu_indices(len(gmat), 1)
            g, t = gmat[iu], tmat[iu]
            gg.extend((g - g.mean()) / (g.std() or 1.))
            tt.extend((t - t.mean()) / (t.std() or 1.))
        return np.array(gg), np.array(tt)

    out = {"per_family": per_family}
    for label, selected in (("pooled", blocks),
                            ("without_germanic", [x for x in blocks if x[0] != "Germanic"])):
        g, t = vectors(selected)
        observed = ev.corr(g, t)
        hits = 0
        rng = np.random.default_rng(seed)
        for _ in range(nperm):
            perms = {f: rng.permutation(len(c)) for f, c, _, _ in selected}
            gp, tp = vectors(selected, permutations=perms)
            hits += abs(ev.corr(gp, tp)) >= abs(observed) - 1e-14
        deleted = []
        for family, codes, _, _ in selected:
            for code in codes:
                dg, dt = vectors(selected, drop=(family, code))
                deleted.append(ev.corr(dg, dt))
        out[label] = dict(n_sources=sum(len(b[1]) for b in selected), n_pairs=len(g),
                          r=observed, rho=float(spearmanr(g, t).statistic),
                          p_taxon=(hits + 1) / (nperm + 1), nperm=nperm,
                          source_deletion_min=min(deleted), source_deletion_max=max(deleted))
    return out


def language_appendix(lang, repl, nperm):
    out = {"descent": family_pairs(lang, language.CONFIGS), "clock": {}, "replication": {}}
    for name, ages in AGES.items():
        out["clock"][name] = clock_block(lang, language.CONFIGS, ages, nperm)
    meta = json.loads(str(repl["meta"]))
    for key, info in meta["models"].items():
        configs = [f"{key}_L{L:02d}" for L in info["layers"]]
        out["replication"][key] = dict(
            descent=family_pairs(repl, configs),
            clock=clock_block(repl, configs, AGES["cited"], nperm), metadata=info)
    out["cross_script"] = language.cross_script(lang)
    # All four originally discussed cross-script cases are shown, not selected
    # after observing percentile stability. Single-configuration and ensemble.
    codes = [str(c) for c in lang["codes"]]
    lookup = {language.short(c): i for i, c in enumerate(codes)}
    ensemble = np.zeros((len(codes), len(codes)))
    for config in language.CONFIGS:
        D = ev.distances(lang[f"glob_{config}"], "euclid")
        ensemble += D / D[np.triu_indices(len(D), 1)].mean()
    ensemble /= len(language.CONFIGS)
    iu = np.triu_indices(len(codes), 1)
    for pair in ("hin-urd", "ben-hin", "kaz-tur", "arb-heb"):
        a, b = pair.split("-")
        d = ensemble[lookup[a], lookup[b]]
        out["cross_script"]["pairs"][pair]["ensemble_euclid_pct"] = float(
            np.mean(ensemble[iu] < d) * 100)
    return out


def synthetic_results(binary):
    Q, rule = ev.quartet_bank(256)
    cells = [("clean", 0., 0.)]
    cells += [(f"radial_{k}", k, 0.) for k in (.5, 1., 1.5, 2.)]
    cells += [(f"mixing_{k}", 0., k) for k in (.1, .2, .35, .5)]
    seeds = []
    for seed in range(5):
        print(f"  synthetic seed {seed}", flush=True)
        leaves, G, paths, depth, Z, rng = gen_latent(8, 64, seed)
        levels = [tuple(paths[i]) for i in leaves]
        _, gold, by_depth = ev.reference(levels)
        deep = by_depth[1] | by_depth[2]
        groups = [p[:-1] for p in levels]
        # Independent streams; draw once and share perturbations across doses.
        donor_rng = np.random.default_rng(seed + 1000)
        radius_rng = np.random.default_rng(seed + 2000)
        donors = np.array([donor_rng.choice([j for j in leaves if paths[j][0] != paths[i][0]])
                           for i in leaves])
        radii = radius_rng.normal(size=len(Z))
        by_cell = {}
        for name, kN, kC in cells:
            X = ((1 - kC) * Z + kC * Z[donors]) * np.exp(kN * radii)[:, None]
            cell = {"radial": kN, "mixing": kC, "metrics": {}}
            for metric in ("euclid", "cosine"):
                D = ev.distances(X, metric)
                row = ev.direct_scores(D, G, groups, Q)
                for method in ("nj", "fastme"):
                    tree = ev.nj(D) if method == "nj" else ev.fastme(D, binary)[0]
                    score = ev.score_tree(tree, gold, G, Q)
                    splits = ev.tree_splits(tree, len(Z))
                    score["deep_recovered"] = len(splits & deep)
                    score["deep_available"] = len(deep)
                    score["deep_split_recall"] = len(splits & deep) / len(deep)
                    # Full split lists are needed for empirical reproducibility;
                    # omit repeated synthetic lists to keep the record compact.
                    score.pop("splits")
                    row[method] = score
                cell["metrics"][metric] = row
            by_cell[name] = cell
        seeds.append(dict(seed=seed, cells=by_cell))
    # Exact radial invariance is a validity gate, not an empirical discovery.
    for item in seeds:
        clean = item["cells"]["clean"]["metrics"]["cosine"]
        for name, cell in item["cells"].items():
            if name.startswith("radial_"):
                row = cell["metrics"]["cosine"]
                for key in ("distance_r", "distance_quartet", "nn_purity"):
                    assert abs(row[key] - clean[key]) < 1e-9
                assert row["nj"]["split_recall"] == clean["nj"]["split_recall"]
    return dict(depth=8, dimension=64, n=256, n_reference_splits=253,
                quartet_rule=rule, seeds=seeds,
                donor_seed_offset=1000, radius_seed_offset=2000,
                order="mix clean leaf features, then multiply by positive radial scale",
                interpretation="Illustrative feature-space perturbation; no fitted encoder or historical inference.")


def archival_results():
    import csv
    out = {"cloud_shape": []}
    # Source extraction, not inferred constants: scalar-channel results are
    # available only as rounded numbers in the original saved SUMMARY.
    summary = (DATA / "supplementary/cloud_magnitude.txt").read_text()
    line = next(x for x in summary.splitlines()
                if x.strip().startswith("partial(shape, gen | CENTROID)"))
    import re
    values = [float(x) for x in re.findall(r"[+-]\d+\.\d+", line)]
    assert len(values) == 2
    for layer, r in zip((8, 12), values):
        out["cloud_shape"].append(dict(channel="paired displacement magnitude", layer=layer, r=r,
            source="data/supplementary/cloud_magnitude.txt", scope="94 within-family pairs; rounded archival estimate"))
    with (DATA / "supplementary/cloud_covariance.csv").open() as f:
        rows = list(csv.DictReader(f))
    for layer in sorted({int(r["layer"]) for r in rows}):
        selected = [r for r in rows if int(r["layer"]) == layer]
        arr = np.array([[float(r[k]) for k in ("bures", "gen", "cent")] for r in selected])
        out["cloud_shape"].append(dict(channel="PCA-40 Bures covariance", layer=layer,
            r=ev.partial(arr[:, 0], arr[:, 1], [arr[:, 2]]), n_pairs=len(arr),
            source="data/supplementary/cloud_covariance.csv", scope="Within-family language association, not topology reconstruction"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastme", type=Path, default=FASTME)
    parser.add_argument("--nperm", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=ROOT / "results/analysis.json")
    args = parser.parse_args()
    for p in INPUTS + [args.fastme]:
        if not p.is_file():
            raise FileNotFoundError(f"Required input missing: {p}")
    if args.nperm != 2000 and args.out == ROOT / "results/analysis.json":
        raise ValueError("Nonstandard permutation counts require a separate output path.")
    versions = {m: importlib.metadata.version(m) for m in ("numpy", "scipy", "lang2vec")}
    version = subprocess.run([str(args.fastme), "-V"], capture_output=True, text=True, check=True)
    res = dict(schema_version=2, generated_at=datetime.now(timezone.utc).isoformat(),
               provenance=dict(python=platform.python_version(), packages=versions,
                    inputs={str(p.relative_to(ROOT)): digest(p) for p in INPUTS},
                    sources={str(p.relative_to(ROOT)): digest(p) for p in SOURCES},
                    analysis_sha256=digest(Path(__file__)),
                    evaluation_sha256=digest(HERE / "evaluation.py"),
                    fastme_version=version.stdout.strip(),
                    fastme_binary_sha256=digest(args.fastme),
                    fastme_source_sha256="d49ff78cca7b76eadf8443efb33f071a7e8a152618ffa3d8e790f167d0a0b176",
                    fastme_options="-m B -n B -s -T 1 -v 0", nperm=args.nperm,
                    statistic="Unique unrooted splits; ties fractional; source-label nulls; adjusted r descriptive"))
    datasets, lang, speech, repl = get_datasets()

    def record(name, fn):
        print(f"[start] {name}", flush=True)
        res[name] = fn()
        (HERE / ".cache").mkdir(exist_ok=True)
        (HERE / ".cache/progress.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
        print(f"[done] {name}", flush=True)

    record("exact_controls", lambda: exact_controls(args.fastme))
    record("domains", lambda: domain_results(datasets, args.fastme, args.nperm))
    record("readouts", lambda: readout_results(lang, speech, datasets, res["domains"], args.fastme))
    record("language_appendix", lambda: language_appendix(lang, repl, args.nperm))
    record("synthetic", lambda: synthetic_results(args.fastme))
    record("archive", archival_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    (HERE / ".cache/progress.json").unlink(missing_ok=True)
    print(f"Complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
