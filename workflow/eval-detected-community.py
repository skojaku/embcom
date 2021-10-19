# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    com_file = snakemake.input["com_file"]
    ref_com_file = snakemake.input["ref_com_file"] if "ref_com_file" in snakemake.input.keys() else None
    K = int(snakemake.params["K"])
    output_file = snakemake.output["output_file"]
else:
    com_file = "../data/communities/ring_of_cliques/community_n=10000_nc=50_cave=50_cdiff=160_sample=1_model=infomap.npz"
    K = int(10000 / 50)
    output_sim_file = ""

# %%
# Evaluation
#
def calc_nmi(y, ypred):
    _, y = np.unique(y, return_inverse=True)
    _, ypred = np.unique(ypred, return_inverse=True)

    Kpred = int(np.max(ypred) + 1)
    K = int(np.max(y) + 1)
    N = len(y)
    U = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N, K))
    Upred = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, Kpred)
    )
    prc = np.array((U.T @ Upred).toarray())
    prc = prc / np.sum(prc)
    pr = np.array(np.sum(prc, axis=1)).reshape(-1)
    pc = np.array(np.sum(prc, axis=0)).reshape(-1)

    # Calculate the mutual information
    Irc = stats.entropy(prc.reshape(-1), np.outer(pr, pc).reshape(-1))

    # Normalize MI
    Q = 2 * Irc / (stats.entropy(pr) + stats.entropy(pc))
    return Q


def calc_esim(y, ypred):
    _, y = np.unique(y, return_inverse=True)
    _, ypred = np.unique(ypred, return_inverse=True)

    Kpred = int(np.max(ypred) + 1)
    K = int(np.max(y) + 1)
    M = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(M, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(M, Kpred)
    )

    fA = np.array(UA.sum(axis=0)).reshape(-1)
    fB = np.array(UB.sum(axis=0)).reshape(-1)
    fAB = (UA.T @ UB).toarray()

    Si = (
        0.5
        * fAB[(y, ypred)]
        * (1.0 / fA[y] + 1.0 / fB[ypred] - np.abs(1.0 / fA[y] - 1.0 / fB[ypred]))
    )
    S = np.mean(Si)
    return S


# %%
# Load community assignment
cids = np.unique(np.load(com_file)["group_ids"], return_inverse=True)[1]
if ref_com_file is not None:
    group_ids = np.unique(np.load(ref_com_file)["group_ids"], return_inverse=True)[1]
else:
    n = int(np.round(len(cids) / K))
    group_ids = np.kron(np.arange(K), np.ones(n)).astype(int)
# %%

# Evaluate
dflist = []
score_esim, score_nmi = calc_esim(group_ids, cids), calc_nmi(group_ids, cids)

dh = pd.DataFrame(
    [
        {"score": score_esim, "score_type": "esim"},
        {"score": score_nmi, "score_type": "nmi"},
    ]
)
dflist += [dh]
res_table = pd.concat(dflist)

# %%
# Save
#
res_table.to_csv(output_file, index=False)
