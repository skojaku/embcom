# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.cluster import normalized_mutual_info_score

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../../../data/multi_partition_model/embedding/n~2500_K~2_cave~10_mu~0.5_sample~0_model_name~modspec_window_length~10_dim~64.npz"
    com_file = "../../../data/multi_partition_model/networks/node_n~2500_K~2_cave~10_mu~0.5_sample~0.npz"
    output_file = ""

from sklearn import cluster

# %%
# Load
#
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
emb[np.isnan(emb)] = 0
emb[np.isinf(emb)] = 0

memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# %% Filter the eigenvectors
def get_focal_eigenvec(emb, focal_eigenvec, pick_multiple_eigenvec, normalize):
    norm_emb = np.linalg.norm(emb, axis=0)
    sorted_emb = emb[:, np.argsort(-norm_emb)]
    if pick_multiple_eigenvec:
        retval = sorted_emb[:, : focal_eigenvec + 1]
    else:
        retval = sorted_emb[:, focal_eigenvec]

    if len(retval.shape) == 1:
        retval = retval.reshape(emb.shape[0], -1)

    if normalize:
        retval = retval / np.linalg.norm(retval, axis=0)
    return retval


#
# Community detection
#
def KMeans(emb, group_ids, metric="euclidean"):
    K = np.max(group_ids) + 1
    if metric == "cosine":
        X = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
    else:
        X = emb
    kmeans = cluster.MiniBatchKMeans(n_clusters=K, random_state=0).fit(X)
    labels = kmeans.labels_
    return labels


def Voronoi(emb, group_ids, metric="euclidean"):
    N = emb.shape[0]
    K = np.max(group_ids) + 1
    U = sparse.csr_matrix(
        (np.ones_like(group_ids), (np.arange(group_ids.size), group_ids)), shape=(N, K)
    )

    # Row normalize
    denom = np.array(U.sum(axis=1)).reshape(-1).astype(float)
    U = sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ U

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


#
# Main
#
results = []
for focal_eigenvec in [0, 1, 2, 3, 4, 5]:
    for pick_multiple_eigenvec in [True, False]:
        for normalize in [True, False]:
            # Filter out the eigenvec
            emb_filtered = get_focal_eigenvec(
                emb, focal_eigenvec, pick_multiple_eigenvec, normalize
            )

            for clus_name in ["kmeans", "voronoi"]:
                for metric in ["cosine", "euclidean"]:

                    # Detect the communities
                    if clus_name == "kmeans":
                        group_ids = KMeans(emb_filtered, memberships, metric=metric)
                    elif clus_name == "voronoi":
                        group_ids = Voronoi(emb_filtered, memberships, metric=metric)

                    # Evaluate the detected communities
                    for scoreType in ["nmi", "esim"]:
                        if scoreType == "nmi":
                            score = normalized_mutual_info_score(memberships, group_ids)
                        elif scoreType == "esim":
                            score = calc_esim(memberships, group_ids)
                        else:
                            raise ValueError("Unknown score type: {}".format(scoreType))

                        results.append(
                            {
                                "score": score,
                                "scoreType": scoreType,
                                "metric": metric,
                                "focal_eigenvec": focal_eigenvec,
                                "clustering": clus_name,
                                "multiple_eigenvec": pick_multiple_eigenvec,
                                "normalize": normalize,
                            }
                        )

df = pd.DataFrame(results)

# %%
#
# Save
#
df.to_csv(output_file, index=False)
