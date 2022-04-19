"""Evaluate the detected communities using the element-centric similarity."""

# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics.cluster import normalized_mutual_info_score

if "snakemake" in sys.modules:
    detected_group_file = snakemake.input["detected_group_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    scoreType = params["scoreType"]
else:
    com_file = "../../data/multi_partition_model/networks/node_n~100000_K~2_cave~50_mu~0.5_sample~0.npz"
    detected_group_file = "../../data/multi_partition_model/communities/clus_n~100000_K~2_cave~50_mu~0.5_sample~0_model_name~leigenmap_window_length~10_dim~0_metric~cosine_clustering~voronoi.npz"
    output_sim_file = "unko"
    scoreType = "nmi"


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

    K = np.maximum(Ka, Kb)
    N = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, K)
    )

    nA = np.array(UA.sum(axis=0)).reshape(-1)
    nB = np.array(UB.sum(axis=0)).reshape(-1)

    nAB = (UA.T @ UB).toarray()
    nAB_rand = np.outer(nA, nB) / N

    # Calc element-centric similarity
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    S = np.sum(np.multiply(Q, (nAB ** 2))) / N

    # Calc the expected element-centric similarity for random partitions
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    Srand = np.sum(np.multiply(Q, (nAB_rand ** 2))) / N
    Scorrected = (S - Srand) / (1 - Srand)
    return Scorrected


if scoreType == "nmi":
    score = normalized_mutual_info_score(memberships, group_ids)
elif scoreType == "esim":
    score = calc_esim(memberships, group_ids)
else:
    raise ValueError("Unknown score type: {}".format(scoreType))
# %%
# Save
#
res_table = pd.DataFrame([{"score": score, "score_type": scoreType}]).to_csv(
    output_file, index=False
)

# %%
