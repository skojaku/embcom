"""Evaluate the detected communities using the element-centric similarity."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    model_name = params["model_name"]

else:
    emb_file = "../../data/multi_partition_model/embedding/n~1000_K~2_cave~50_mu~0.70_sample~1_model_name~leigenmap_window_length~10_dim~64.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~2500_K~2_cave~50_mu~0.70_sample~1.npz"
    model_name = "leigenmap"
    output_file = "unko"
    metric = "cosine"


from sklearn import cluster


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


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
emb[np.isnan(emb)] = 0
emb[np.isinf(emb)] = 0
memberships = pd.read_csv(com_file)["membership"].values.astype(int)
emb_copy = emb.copy()

# Normalize the eigenvector by dimensions
results = {}
for dimThreshold in [True, False]:
    for normalize in [True, False]:
        emb = emb_copy.copy()
        if model_name == "nonbacktracking":
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            idx = np.argmax(norm)
            threshold = np.sqrt(norm[idx])
            keep = norm >= threshold
            keep[idx] = False
            emb = emb[:, keep]

        if normalize:
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            emb = np.einsum("ij,j->ij", emb, 1 / np.maximum(norm, 1e-32))

        # Evaluate
        group_ids = KMeans(emb, memberships, metric=metric)

        key = f"normalize~{normalize}_dimThreshold~{dimThreshold}"
        results[key] = group_ids

# %%
# Save
#
np.savez(output_file, **results)
