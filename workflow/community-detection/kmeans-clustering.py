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
else:
    emb_file = "../../data/multi_partition_model/embedding/n~100000_K~32_cave~50_mu~0.95_sample~7_model_name~node2vec_window_length~10_dim~64.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~100000_K~32_cave~50_mu~0.95_sample~7.npz"
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


from sklearn import cluster


def KMeans(emb, group_ids, metric="euclidean"):
    K = np.max(group_ids) + 1
    if metric == "cosine":
        X = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
    else:
        X = emb
    kmeans = cluster.MiniBatchKMeans(n_clusters=K, random_state=0).fit(X)
    labels = kmeans.labels_
    return labels


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
emb[np.isnan(emb)] = 0
emb[np.isinf(emb)] = 0

memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# Evaluate
group_ids = KMeans(emb, memberships, metric=metric)

# %%
# Save
#
np.savez(output_file, group_ids=group_ids)
