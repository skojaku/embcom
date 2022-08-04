"""Evaluate the detected communities using the element-centric similarity."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    normalize = params["normalize"]
    dimThreshold = params["dimThreshold"]
    model_name = params["model_name"]

    if isinstance(normalize, str):
        normalize == normalize == "True"

    if isinstance(dimThreshold, str):
        dimThreshold == dimThreshold == "True"
else:
    emb_file = "../../data/multi_partition_model/embedding/n~100000_K~2_cave~50_mu~0.5_sample~0_model_name~leigenmap_window_length~10_dim~0.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~100000_K~2_cave~50_mu~0.5_sample~0.npz"
    output_file = "unko"
    metric = "cosine"


def row_normalize(mat, mode="prob"):
    """Normalize a sparse CSR matrix row-wise (each row sums to 1) If a row is
    all 0's, it remains all 0's.

    Parameters
    ----------
    mat : scipy.sparse.csr matrix
        Matrix in CSR sparse format
    Returns
    -------
    out : scipy.sparse.csr matrix
        Normalized matrix in CSR sparse format
    """
    if mode == "prob":
        denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
        return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat
    elif mode == "norm":
        denom = np.sqrt(np.array(mat.multiply(mat).sum(axis=1)).reshape(-1))
        return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat
    return np.nan


def KMeans(emb, group_ids, metric="euclidean"):
    N = emb.shape[0]
    K = np.max(group_ids) + 1
    U = sparse.csr_matrix(
        (np.ones_like(group_ids), (np.arange(group_ids.size), group_ids)), shape=(N, K)
    )
    U = row_normalize(U)
    centers = U.T @ emb
    if metric == "cosine":
        nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
        ncenters = np.einsum("ij,i->ij", centers, 1 / np.linalg.norm(centers, axis=1))
        return np.argmax(nemb @ ncenters.T, axis=1)
    elif metric == "dotsim":
        return np.argmax(emb @ centers.T, axis=1)
    elif metric == "euclidean":
        norm_emb = np.linalg.norm(emb, axis=1) ** 2
        norm_cent = np.linalg.norm(centers, axis=1) ** 2
        dist = np.add.outer(norm_emb, norm_cent) - 2 * emb @ centers.T
        return np.argmin(dist, axis=1)


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
emb[np.isnan(emb)] = 0
emb[np.isinf(emb)] = 0

# Normalize the eigenvector by dimensions
if dimThreshold:
    if model_name == "nonbacktracking":
        norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
        idx = np.argmax(norm)
        threshold = np.sqrt(norm[idx])
        keep = norm >= threshold
        keep[idx] = False
        emb = emb[:, keep]

if normalize:
    norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
    emb = np.einsum("ij,j->ij", emb, 1 / np.maximum(norm, 1e-32))

memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# Evaluate
X = emb.copy()
if metric == "cosine":
    X = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
group_ids = KMeans(X, memberships, metric=metric)
# %%
# Save
#
np.savez(output_file, group_ids=group_ids)
