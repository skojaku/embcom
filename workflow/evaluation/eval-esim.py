"""Evaluate the detected communities using the element-centric similarity."""
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    detected_group_file = snakemake.input["detected_group_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/embeddings/two_coms/embeddings/xxx"
    K = 2
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
    _, y = np.unique(y, return_inverse=True)
    _, ypred = np.unique(ypred, return_inverse=True)

    K = len(set(y))
    M = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(M, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(M, K)
    )

    fA = np.array(UA.sum(axis=0)).reshape(-1)
    fB = np.array(UB.sum(axis=0)).reshape(-1)

    # fAB = UA.T @ UB
    # Si = (
    #    0.5
    #    * np.array(fAB[(y, ypred)]).reshape(-1)
    #    * (1.0 / fA[y] + 1.0 / fB[ypred] - np.abs(1.0 / fA[y] - 1.0 / fB[ypred]))
    # )
    # S = np.mean(Si)
    ids, freq = np.unique(y * K + ypred, return_counts=True)
    y, ypred = divmod(ids, K)
    S = 0
    UAT = sparse.csr_matrix(UA.T)
    UBT = sparse.csr_matrix(UB.T)
    for i in range(len(y)):
        fab = UAT[y[i], :].multiply(UBT[ypred[i], :]).sum()
        S += (
            0.5
            * freq[i]
            * fab
            * (
                1.0 / fA[y[i]]
                + 1.0 / fB[ypred[i]]
                - np.abs(1.0 / fA[y[i]] - 1.0 / fB[ypred[i]])
            )
        )
    S /= M
    return S


score_esim = calc_esim(group_ids, group_ids)

#
# Save
#
res_table = pd.DataFrame([{"score": score_esim, "score_type": "esim"}]).to_csv(
    output_file, index=False
)
