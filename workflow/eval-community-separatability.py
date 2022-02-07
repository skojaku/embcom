""""Calculate the distance between nodes between and within the ground-truth
communities."""
# %%
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = (
        snakemake.input["com_file"] if "com_file" in snakemake.input.keys() else None
    )
    K = int(snakemake.params["K"])
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/embeddings/lfr/embnet_n=10000_k=20_tau=2_tau2=3_mu=0.25_maxk=100_maxc=250_sample=1_model=node2vec_wl=10_dim=64.npz"
    com_file = "../data/networks/lfr/community_n=10000_k=20_tau=2_tau2=3_mu=0.25_maxk=100_maxc=250_sample=1.npz"
    K = 10
    output_file = "tmp.csv"


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
    K = len(np.unique(group_ids))

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import sparse


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


clf = LinearDiscriminantAnalysis()
clf.fit(emb, group_ids)
y_pred = clf.predict(emb)
esim = calc_esim(group_ids, y_pred)

# %%
pd.DataFrame([{"score": esim}]).to_csv(output_file, index=False)
