"""Brownian features on a balanced tree, preserving the published random draws."""
import numpy as np

from .evaluation import reference


def gen_latent(depth, dimension, seed):
    """Return leaves, tree distances, paths, depth, features and RNG state.

    Edges are first visited in leaf/prefix order. Changing this order changes
    the mapping from random draws to edges and hence the five published seeds.
    Expected squared feature distance / dimension equals the tree distance;
    an observed feature-distance matrix is not exactly additive.
    """
    rng = np.random.default_rng(seed)
    leaves = list(range(2 ** depth))
    paths = {i: tuple((i >> k) & 1 for k in reversed(range(depth))) for i in leaves}
    G, _, _ = reference(list(paths.values()))
    increments = {}
    Z = np.zeros((len(leaves), dimension))
    for i in leaves:
        for k in range(depth):
            edge = paths[i][:k + 1]
            if edge not in increments:
                increments[edge] = rng.normal(0, 1.0, size=dimension)
            Z[i] = Z[i] + increments[edge]
    return leaves, G.astype(np.float32), paths, depth, Z, rng
