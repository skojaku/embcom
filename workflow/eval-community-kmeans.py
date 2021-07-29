# %%
import sys
import pathlib
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn import metrics
from tqdm import tqdm
import faiss
from scipy import stats
from sklearn.cluster import KMeans

if "snakemake" in sys.modules:
    emb_files = snakemake.input["emb_files"]
    output_sim_file = snakemake.output["output_sim_file"]
else:
    emb_files = list(glob.glob("../data/embeddings/two_coms/embeddings/*"))
    output_sim_file = "../data/results/two_coms/res-kmeans.csv"

# %%
# Load
#
def get_params(filename):
    params = pathlib.Path(filename).stem.split("_")
    retval = {"filename": filename}
    for p in params:
        if "=" not in p:
            continue
        kv = p.split("=")
        retval[kv[0]] = kv[1]
    return retval


emb_file_table = pd.DataFrame([get_params(r) for r in emb_files])
emb_file_table = emb_file_table.rename(
    columns={"filename": "emb_file", "id": "param_id"}
)

# %%
# Evaluation
#
def calc_nmi(y, ypred):
    _, y = np.unique(y, return_inverse=True)
    _, ypred = np.unique(ypred, return_inverse=True)

    K = len(set(y))
    N = len(y)
    U = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N, K))
    Upred = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, K)
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

    K = len(set(y))
    M = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(M, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(M, K)
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


def eval(emb, group_ids, K=2, iterations=5, metric="cosine"):

    _, group_ids = np.unique(group_ids, return_inverse=True)

    emb[np.isnan(emb)] = 0
    emb[np.isinf(emb)] = 0
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=1, max_iter = 100).fit(emb)
    cids = kmeans.labels_
    return calc_esim(group_ids, cids), calc_nmi(group_ids, cids)

    results = []
    if metric == "cosine":
        km = faiss.Kmeans(emb.shape[1], K, niter=iterations, spherical=True)
    else:
        km = faiss.Kmeans(emb.shape[1], K, niter=iterations)
    km.train(emb)
    _, cids = km.index.search(emb, 1)
    return calc_esim(group_ids, cids), calc_nmi(group_ids, cids)


def eval_clu(df):

    # Load emebdding
    emb_list = {}
    for _i, row in df.iterrows():
        emb = np.load(row["emb_file"])["emb"]
        emb = emb.copy(order="C").astype(np.float32)
        emb_list[row["emb_file"]] = emb

    # Evaluate
    dflist = []
    for metric in ["cosine", "euclidean"]:
        for _i, row in df.copy().iterrows():
            emb = emb_list[row["emb_file"]]
            X = emb.copy()
            if metric == "cosine":
                X = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
            n = int(X.shape[0] / 2)
            y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
            score_esim, score_nmi = eval(X, y, iterations=1, metric=metric)

            dh = pd.DataFrame(
                [
                    {"score": score_esim, "metric": metric, "score_type": "esim"},
                    {"score": score_nmi, "metric": metric, "score_type": "nmi"},
                ]
            )
            for k, v in row.items():
                dh[k] = v

            dflist += [dh]
    dh = pd.concat(dflist)
    return dh


list_results = Parallel(n_jobs=30)(
    delayed(eval_clu)(df) for emb_file, df in tqdm(emb_file_table.groupby("emb_file"))
)

# %%
# Preprocess
#
sim_table = pd.concat(list_results)

# %%
# Save
#
sim_table.to_csv(output_sim_file, index=False)

# %%
