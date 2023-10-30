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
    model_name = params["model_name"]

else:

    emb_file = "../../data/lfr/embedding/n~1000_k~100_tau~3_tau2~1_minc~50_mu~0.70_sample~9_model_name~linearized-node2vec_window_length~10_dim~64.npz"
    com_file = "../../data/lfr/networks/node_n~1000_k~100_tau~3_tau2~1_minc~50_mu~0.70_sample~9.npz"
    model_name = "linearized-node2vec"
    output_file = "unko"
    metric = "cosine"


# %%


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

from sklearn.cluster import HDBSCAN

def KMeans(emb, group_ids, metric="euclidean"):
    N = emb.shape[0]
    K = np.max(group_ids) + 1
    hdb = HDBSCAN()
    if metric == "cosine":
        nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
        hdb.fit(nemb)
    else:
        hdb.fit(emb)
        labels = hdb.labels_ 
    return labels


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)
print(emb.shape)
# Remove nan embedding
remove = np.isnan(np.array(np.sum(emb, axis=1)).reshape(-1))
keep = np.where(~remove)[0]
n_nodes = emb.shape[0]
emb = emb[keep, :]
memberships = memberships[keep]

emb_copy = emb.copy()

results = {}
for dimThreshold in [True, False]:
    for normalize in [True, False]:
        emb = emb_copy.copy()
        if model_name == "nonbacktracking":
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            idx = np.argmax(norm)
            threshold = np.sqrt(norm[idx])
            keep_dims = norm >= threshold
            keep_dims[idx] = False
            if any(keep_dims) is False:
                keep_dims[idx] = True
            emb = emb[:, keep_dims]

        if normalize:
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            emb = np.einsum("ij,j->ij", emb, 1 / np.maximum(norm, 1e-32))

        # Evaluate
        group_ids = KMeans(emb, memberships, metric=metric)

        key = f"normalize~{normalize}_dimThreshold~{dimThreshold}"

        # Set the group index to nan if the node has no embedding
        group_ids_ = np.zeros(n_nodes) * np.nan
        group_ids_[keep] = group_ids
        results[key] = group_ids_

# %%
# Save
#
np.savez(output_file, **results)
