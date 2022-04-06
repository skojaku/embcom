"""Evaluate the detected communities using the element-centric similarity."""

# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    detected_group_file = snakemake.input["detected_group_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
else:
    com_file = "../../data/multi_partition_model/networks/node_n~2500_K~2_cave~20_mu~0.5_sample~0.npz"
    detected_group_file = "../../data/multi_partition_model/communities/clus_n~2500_K~2_cave~20_mu~0.5_sample~0_model_name~node2vec-matrixfact-limit_window_length~10_dim~0_metric~cosine_clustering~kmeans.npz"
    output_sim_file = "unko"


#
# Load
#
memberships = pd.read_csv(com_file)["membership"].values.astype(int)
group_ids = np.load(detected_group_file)["group_ids"]

#
# Evaluation
#
def calc_esim(y, ypred):
    """Element centric similarity."""
    ylab, y = np.unique(y, return_inverse=True)
    ypredlab, ypred = np.unique(ypred, return_inverse=True)

    Ka, Kb = len(ylab), len(ypredlab)

    K = int(np.maximum(np.max(y), np.max(ypred))) + 1
    M = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(M, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(M, K)
    )

    fA = np.array(UA.sum(axis=0)).reshape(-1)
    fB = np.array(UB.sum(axis=0)).reshape(-1)

    fAB = UA.T @ UB

    S = 0
    for i in range(len(y)):
        S += (fAB[y[i], ypred[i]] ** 2) * np.minimum(
            1 / np.maximum(1, fA[y[i]]), 1 / np.maximum(1, fB[ypred[i]])
        )
    S /= M

    Srand = np.minimum(1 / Ka, 1 / Kb)

    Scorrected = (S - Srand) / (1 - Srand)
    return Scorrected


score_esim = calc_esim(memberships, group_ids)

# %%
# Save
#
res_table = pd.DataFrame([{"score": score_esim, "score_type": "esim"}]).to_csv(
    output_file, index=False
)

# %%
