"""Evaluate the detected communities using the element-centric similarity."""
import sys

import numpy as np
import pandas as pd
import utils
from scipy import sparse, stats

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    metric = "cosine"
else:
    emb_file = "../data/embeddings/two_coms/embeddings/xxx"
    output_file = "unko"
    metric = "cosine"


def KMeans(emb, group_ids, metric="euclidean"):
    N = emb.shape[0]
    K = np.max(group_ids) + 1
    U = sparse.csr_matrix(
        (np.ones_like(group_ids), (np.arange(group_ids.size), group_ids)), shape=(N, K)
    )
    U = utils.row_normalize(U)
    centers = U.T @ emb
    if metric == "cosine":
        nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
        ncenters = np.einsum("ij,i->ij", centers, 1 / np.linalg.norm(centers, axis=1))
        return np.argmax(nemb @ ncenters.T, axis=1)
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

memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# Evaluate
dflist = []
# for metric in ["cosine", "euclidean"]:
X = emb.copy()
if metric == "cosine":
    X = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
group_ids = KMeans(emb, memberships, metric=metric)

# %%
# Save
#
np.savez(output_file, group_ids=group_ids)
