"""Shared, explicit evaluation of distances and unrooted reference splits."""
from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from scipy.spatial.distance import pdist, squareform

TOL = 1e-10


def validate_distance(D):
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or len(D) < 4:
        raise ValueError("Distances must be square with at least four sources.")
    if not np.isfinite(D).all():
        raise ValueError("Nonfinite distances.")
    tol = TOL * max(1.0, float(np.max(np.abs(D))))
    if np.max(np.abs(D - D.T)) > tol or np.max(np.abs(np.diag(D))) > tol:
        raise ValueError("Distances must be symmetric with a zero diagonal.")
    if D.min() < -tol:
        raise ValueError("Negative input dissimilarity.")
    D = np.maximum((D + D.T) / 2, 0)
    np.fill_diagonal(D, 0)
    return D


def distances(X, metric):
    X = np.asarray(X, np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Expected a finite source-by-feature array.")
    if metric == "euclid":
        D = squareform(pdist(X))
    elif metric == "cosine":
        norms = np.linalg.norm(X, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("Cosine is undefined for a zero source vector.")
        Y = X / norms[:, None]
        D = np.clip(1 - Y @ Y.T, 0, 2)
        np.fill_diagonal(D, 0)
    else:
        raise ValueError(metric)
    return validate_distance(D)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    scale = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / scale) if scale > 1e-14 else 0.0


def partial(y, x, controls):
    C = np.column_stack([np.ones(len(y))] + list(controls))
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    return corr(ry, rx)


def canonical_split(side, n):
    side = frozenset(side)
    other = frozenset(range(n)) - side
    if min(len(side), len(other)) < 2:
        return None
    return min((side, other), key=lambda s: (len(s), tuple(sorted(s))))


def reference(levels):
    """Unit-length prefix hierarchy. Repeated unary levels affect G, not splits."""
    n, depth = len(levels), len(levels[0])
    if any(len(x) != depth for x in levels) or len(set(levels)) != n:
        raise ValueError("Reference paths need equal depths and unique leaf identities.")
    G = np.zeros((n, n), float)
    for i, j in combinations(range(n), 2):
        shared = 0
        while shared < depth and levels[i][shared] == levels[j][shared]:
            shared += 1
        G[i, j] = G[j, i] = 2 * (depth - shared)
    by_depth = {}
    for d in range(1, depth):
        groups = {}
        for i, path in enumerate(levels):
            groups.setdefault(path[:d], set()).add(i)
        by_depth[d] = {s for g in groups.values()
                       if (s := canonical_split(g, n)) is not None}
    return G, set().union(*by_depth.values()), by_depth


def nj(D, labels=None):
    """Standard NJ Q criterion and reduction; returns topology, no rooting claim."""
    D = validate_distance(D)
    active = list(range(len(D))) if labels is None else list(labels)
    while len(active) > 2:
        m = len(active)
        sums = D.sum(axis=1)
        Q = (m - 2) * D - sums[:, None] - sums[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = sorted(np.unravel_index(np.argmin(Q), Q.shape))
        keep = np.ones(m, bool)
        keep[[i, j]] = False
        reduced = .5 * (D[i, keep] + D[j, keep] - D[i, j])
        next_D = np.zeros((m - 1, m - 1))
        next_D[:-1, :-1] = D[np.ix_(keep, keep)]
        next_D[-1, :-1] = next_D[:-1, -1] = reduced
        active = [x for k, x in enumerate(active) if keep[k]] + [(active[i], active[j])]
        D = next_D
    return tuple(active)


def tree_splits(tree, n):
    leaves = []
    splits = set()

    def visit(node):
        if isinstance(node, (int, np.integer)):
            leaves.append(int(node))
            return {int(node)}
        below = set()
        for child in node:
            side = visit(child)
            if (s := canonical_split(side, n)) is not None:
                splits.add(s)
            below.update(side)
        return below

    visit(tree)
    if sorted(leaves) != list(range(n)):
        raise ValueError("Tree leaves do not match distance rows exactly.")
    return splits


def topology_distances(splits, n):
    """Unit internal edges plus leaf edges; independent of rooting/branch lengths."""
    D = np.full((n, n), 2.0)
    for side in splits:
        mask = np.array([i in side for i in range(n)])
        D += mask[:, None] != mask[None, :]
    np.fill_diagonal(D, 0)
    return D


def parse_newick(text):
    """Read integer tNNNN leaf labels from FastME; ignore lengths and node labels."""
    tokens = re.findall(r"[(),:;]|[^\s(),:;]+", text)
    pos = 0

    def node():
        nonlocal pos
        if tokens[pos] == "(":
            pos += 1
            children = [node()]
            while tokens[pos] == ",":
                pos += 1
                children.append(node())
            if tokens[pos] != ")":
                raise ValueError("Invalid Newick closing token.")
            pos += 1
            if tokens[pos] not in (":", ",", ")", ";"):
                pos += 1
            result = tuple(children)
        else:
            label = tokens[pos].strip("'\"")
            if not re.fullmatch(r"t\d+", label):
                raise ValueError(f"Unexpected FastME label: {label}")
            result = int(label[1:])
            pos += 1
        if tokens[pos] == ":":
            pos += 1
            float(tokens[pos])  # validate, including negative or scientific notation
            pos += 1
        return result

    tree = node()
    if tokens[pos:] != [";"]:
        raise ValueError("Trailing or incomplete Newick.")
    return tree


def fastme(D, binary):
    D = validate_distance(D)
    with tempfile.TemporaryDirectory(prefix="phylogeny-fastme-") as tmp:
        tmp = Path(tmp)
        matrix, output = tmp / "input.phy", tmp / "tree.nwk"
        matrix.write_text(str(len(D)) + "\n" + "\n".join(
            # FastME's input validator rejects scientific notation. Positional
            # shortest-round-trip decimals preserve even near-zero distances.
            f"t{i:04d} " + " ".join(np.format_float_positional(v, unique=True) for v in row)
            for i, row in enumerate(D)) + "\n")
        command = [str(binary), "-i", str(matrix), "-o", str(output),
                   "-I", str(tmp / "info.txt"), "-m", "B", "-n", "B", "-s",
                   "-T", "1", "-v", "0"]
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if process.returncode or not output.exists():
            raise RuntimeError(f"FastME failed: {process.stdout}\n{process.stderr}")
        newick = output.read_text().strip()
        tree = parse_newick(newick)
        tree_splits(tree, len(D))  # verify row-to-label round trip
        return tree, newick


def quartet_bank(n, count=40000, seed=0):
    if n <= 30 or comb(n, 4) <= count:
        return np.array(list(combinations(range(n), 4)), dtype=np.int32), "exhaustive"
    rng = np.random.default_rng(seed)
    seen, rows = set(), []
    while len(rows) < count:
        q = tuple(sorted(rng.choice(n, 4, replace=False).tolist()))
        if q not in seen:
            seen.add(q)
            rows.append(q)
    return np.array(rows, dtype=np.int32), "uniform distinct sample"


def quartet_sums(D, Q):
    i, j, k, l = Q.T
    return np.column_stack((D[i, j] + D[k, l],
                            D[i, k] + D[j, l],
                            D[i, l] + D[j, k]))


def minimum_mask(sums, scale):
    return np.abs(sums - sums.min(axis=1, keepdims=True)) <= TOL * max(1., scale)


def resolved_reference(G, Q):
    mins = minimum_mask(quartet_sums(G, Q), float(G.max()))
    resolved = mins.sum(axis=1) == 1
    return Q[resolved], np.argmax(mins[resolved], axis=1)


def quartet_credit(D, Q, answers):
    if len(Q) == 0:
        return None
    tied = minimum_mask(quartet_sums(D, Q), float(D.max()))
    return float(np.mean(tied[np.arange(len(Q)), answers] / tied.sum(axis=1)))


def neighbor_weights(D):
    X = np.array(D, copy=True)
    np.fill_diagonal(X, np.inf)
    hit = np.abs(X - X.min(axis=1, keepdims=True)) <= TOL * max(1., float(D.max()))
    return hit / hit.sum(axis=1, keepdims=True)


def group_matrix(groups):
    return np.array([[a == b for b in groups] for a in groups], dtype=bool)


def nn_purity(D, groups):
    return float((neighbor_weights(D) * group_matrix(groups)).sum(axis=1).mean())


def nn_chance(groups):
    same = group_matrix(groups)
    return float((same.sum() - len(same)) / (len(same) * (len(same) - 1)))


def direct_scores(D, G, groups, Q):
    Qr, answers = resolved_reference(G, Q)
    iu = np.triu_indices(len(D), 1)
    sums = np.sort(quartet_sums(D, Q), axis=1)
    residual = (sums[:, 2] - sums[:, 1]) / (2 * max(float(D.max()), 1e-12))
    return dict(distance_r=corr(D[iu], G[iu]),
                nn_purity=nn_purity(D, groups), nn_chance=nn_chance(groups),
                distance_quartet=quartet_credit(D, Qr, answers),
                n_quartets=len(Q), n_resolved=len(Qr),
                four_point_mean=float(residual.mean()),
                four_point_sample_max=float(residual.max()))


def score_tree(tree, gold_splits, G, Q):
    got = tree_splits(tree, len(G))
    Qr, answers = resolved_reference(G, Q)
    return dict(recovered=len(got & gold_splits), available=len(gold_splits),
                split_recall=len(got & gold_splits) / len(gold_splits) if gold_splits else None,
                tree_quartet=quartet_credit(topology_distances(got, len(G)), Qr, answers),
                reference_fully_resolved=(len(gold_splits) == len(G) - 3),
                full_tree_exact=(got == gold_splits) if len(gold_splits) == len(G) - 3 else None,
                normalized_rf=len(got ^ gold_splits) / (2 * (len(G) - 3))
                if len(gold_splits) == len(G) - 3 else None,
                splits=[sorted(s) for s in sorted(got, key=lambda s: (len(s), tuple(sorted(s))))])


def permutation_tests(D, G, groups, Q, nperm=2000, seed=0):
    """Randomize source identities of D, holding reference and group sizes fixed."""
    n = len(D)
    iu = np.triu_indices(n, 1)
    target = G[iu] - G[iu].mean()
    x = D[iu] - D[iu].mean()
    denom = np.linalg.norm(target) * np.linalg.norm(x)
    obs_r = corr(D[iu], G[iu])
    Qr, answers = resolved_reference(G, Q)
    obs_q = quartet_credit(D, Qr, answers)
    weights, same = neighbor_weights(D), group_matrix(groups)
    obs_nn = float((weights * same).sum() / n)
    hits = np.zeros(3, int)
    rng = np.random.default_rng(seed)
    null_sum = np.zeros(3)
    for _ in range(nperm):
        p = rng.permutation(n)
        Dp = D[np.ix_(p, p)]
        r = float(Dp[iu] @ target / denom) if denom > 1e-14 else 0.
        q = quartet_credit(Dp, Qr, answers) if obs_q is not None else 0.
        purity = float((weights[np.ix_(p, p)] * same).sum() / n)
        hits += [abs(r) >= abs(obs_r) - 1e-14,
                 q >= obs_q - 1e-14 if obs_q is not None else True,
                 purity >= obs_nn - 1e-14]
        null_sum += [r, q, purity]
    pvals = (hits + 1) / (nperm + 1)
    return dict(nperm=nperm, seed=seed, distance_r_p=float(pvals[0]),
                distance_quartet_p=float(pvals[1]) if obs_q is not None else None,
                nn_purity_p=float(pvals[2]),
                null_means=dict(zip(("distance_r", "distance_quartet", "nn_purity"),
                                    (null_sum / nperm).tolist())),
                null="Uniform taxon relabelling; r two-sided, agreement and purity upper-tail.")
