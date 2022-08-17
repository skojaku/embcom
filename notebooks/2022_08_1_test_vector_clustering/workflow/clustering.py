# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score
from Louvain4Vectors import Louvain4Vectors
from sklearn import cluster

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    node_table_file = snakemake.input["node_table_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"


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
    S = np.sum(np.multiply(Q, (nAB**2))) / N

    # Calc the expected element-centric similarity for random partitions
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    Srand = np.sum(np.multiply(Q, (nAB_rand**2))) / N
    Scorrected = (S - Srand) / (1 - Srand)
    return Scorrected


def KMeans(emb, group_ids, metric="euclidean"):
    K = np.max(group_ids) + 1
    if metric == "cosine":
        X = np.einsum(
            "ij,i->ij", emb, 1 / np.maximum(np.linalg.norm(emb, axis=1), 1e-24)
        )
        X = emb
    kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(X)
    # kmeans = cluster.MiniBatchKMeans(n_clusters=K, random_state=0).fit(X)
    # kmeans = cluster.MiniBatchKMeans(n_clusters=K, random_state=0).fit(X)
    return kmeans.labels_


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


def Voronoi(emb, group_ids, metric="euclidean"):
    N = emb.shape[0]
    K = np.max(group_ids) + 1
    U = sparse.csr_matrix(
        (np.ones_like(group_ids), (np.arange(group_ids.size), group_ids)), shape=(N, K)
    )
    U = row_normalize(U)
    centers = U.T @ emb
    if metric == "cosine":
        nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
        ncenters = np.einsum("ij,i->ij", centers, 1 / np.linalg.norm(centers, axis=1))
        return np.argmax(nemb @ ncenters.T, axis=1)
    elif metric == "dotsim":
        return np.argmax(emb @ centers.T, axis=1)
    elif metric == "euclidean":
        norm_emb = np.linalg.norm(emb, axis=1) ** 2
        norm_cent = np.linalg.norm(centers, axis=1) ** 2
        dist = np.add.outer(norm_emb, norm_cent) - 2 * emb @ centers.T
        return np.argmin(dist, axis=1)


#
# Load
#
emb = np.load(emb_file)["emb"]
node_table = pd.read_csv(node_table_file)

# Clustering
metric = "cosine"
results = []
for model_name in ["kmeans", "voronoi"]:
    for normalize in [True, False]:
        if normalize:
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            X = np.einsum("ij,j->ij", emb, 1 / np.maximum(norm, 1e-32))

        # Evaluate

        if model_name == "kmeans":
            cids = KMeans(emb, node_table["membership"], metric=metric)
        elif model_name == "voronoi":
            cids = Voronoi(emb, node_table["membership"], metric=metric)

        key = f"clustering~{model_name}_normalize~{normalize}"

        # Evaluation
        esim_score = calc_esim(node_table["membership"], cids)
        nmi_score = normalized_mutual_info_score(node_table["membership"], cids)
        results += [
            {"score": esim_score, "score_type": "esim", "clustering": key},
            {"score": nmi_score, "score_type": "nmi", "clustering": key},
        ]

# Clustering
model = Louvain4Vectors()
cids = model.clustering(emb)


# Evaluation
esim_score = calc_esim(node_table["membership"], cids)
nmi_score = normalized_mutual_info_score(node_table["membership"], cids)
results += [
    {"score": esim_score, "score_type": "esim", "clustering": "louvain"},
    {"score": nmi_score, "score_type": "nmi", "clustering": "louvain"},
]
res_table = pd.DataFrame(results).to_csv(output_file, index=False)
