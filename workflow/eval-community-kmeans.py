# %%
import sys

import numpy as np
import pandas as pd
import utils
from scipy import sparse, stats

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = (
        snakemake.input["com_file"] if "com_file" in snakemake.input.keys() else None
    )
    K = int(snakemake.params["K"])
    output_sim_file = snakemake.output["output_sim_file"]

else:
    emb_file = "../data/embeddings/two_coms/embeddings/xxx"
    K = 2
    output_sim_file = "unko"

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
    prc = U.T @ Upred
    prc = prc / prc.sum()
    pr = np.array(prc.sum(axis=1)).reshape(-1)
    pc = np.array(prc.sum(axis=0)).reshape(-1)

    # Calculate the mutual information
    r, c, v = sparse.find(prc)
    Irc = -np.sum(v * np.log(v / np.maximum(pr[r] * pc[c], 1e-32)))
    # Irc = stats.entropy(prc.reshape(-1), np.outer(pr, pc).reshape(-1))

    # Normalize MI
    Q = 2 * Irc / (stats.entropy(pr) + stats.entropy(pc))
    return Q


# def calc_nmi(y, ypred):
#    _, y = np.unique(y, return_inverse=True)
#    _, ypred = np.unique(ypred, return_inverse=True)
#
#    K = len(set(y))
#    N = len(y)
#    U = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N, K))
#    Upred = sparse.csr_matrix(
#        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, K)
#    )
#    prc = U.T @ Upred
#    prc.data = prc.data / np.sum(prc.data)
#    pr = np.array(prc.sum(axis=1)).reshape(-1)
#    pc = np.array(prc.sum(axis=0)).reshape(-1)
#
#    # Calculate the mutual information
#    r, c, v = sparse.find(prc)
#    Irc = stats.entropy(v, pr[r] * pc[c])
#
#    # Normalize MI
#    Q = 2 * Irc / (stats.entropy(pr) + stats.entropy(pc))
#    return Q


def calc_esim(y, ypred):
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

if com_file is None:
    n = int(np.round(emb.shape[0] / K))
    group_ids = np.kron(np.arange(K), np.ones(n)).astype(int)
else:
    group_ids = np.load(com_file)["group_ids"]

# Evaluate
dflist = []
for metric in ["cosine"]:
    # for metric in ["cosine", "euclidean"]:
    X = emb.copy()
    if metric == "cosine":
        X = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
    n = int(X.shape[0] / 2)

    cids = KMeans(emb, group_ids, metric=metric)
    # score_esim = calc_esim(group_ids, cids)
    # score_nmi = -1
    score_esim, score_nmi = calc_esim(group_ids, cids), calc_nmi(group_ids, cids)

    dh = pd.DataFrame(
        [
            {"score": score_esim, "metric": metric, "score_type": "esim"},
            {"score": score_nmi, "metric": metric, "score_type": "nmi"},
        ]
    )
    dflist += [dh]
res_table = pd.concat(dflist)

# %%
# Save
#
res_table.to_csv(output_sim_file, index=False)
