"""Evaluate the detected communities using the element-centric similarity."""

# %%
import sys
import embcom
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import normalized_mutual_info_score

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    net_file = snakemake.input["net_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    model_name = params["model_name"]

else:
    emb_file = "../../data/empirical/embedding/netdata~polblog_model_name~modspec_window_length~10_dim~64.npz"
    com_file = "../../data/empirical/networks/node_netdata~polblog.npz"
    net_file = "../../data/empirical/networks/net_netdata~polblog.npz"
    model_name = "modspec"
    output_file = "unko"
    metric = "cosine"


from sklearn import cluster


def KMeans(emb, group_ids, metric="euclidean"):
    K = np.max(group_ids) + 1
    if (metric == "cosine") & (emb.shape[1] > 1):
        X = np.einsum(
            "ij,i->ij", emb, 1 / np.maximum(np.linalg.norm(emb, axis=1), 1e-24)
        )
    else:
        X = emb.copy()
    kmeans = cluster.KMeans(n_clusters=K).fit(X)
    # kmeans = cluster.MiniBatchKMeans(n_clusters=K, random_state=0).fit(X)
    # kmeans = cluster.MiniBatchKMeans(n_clusters=K, random_state=0).fit(X)
    return kmeans.labels_


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)
A = sparse.load_npz(net_file)

emb = np.nan_to_num(emb)
emb_copy = emb.copy()

if model_name in [
    "leigenmap",
    "modspec",
    "modspec2",
    "linearized-node2vec",
    "nonbacktracking",
]:
    K = np.max(memberships) + 1  # proposed framework
    if model_name == "nonbacktracking":
        model = embcom.NonBacktrackingSpectralEmbedding()
    elif model_name == "leigenmap":
        model = embcom.LaplacianEigenMap()
    elif model_name == "linearized-node2vec":
        model = embcom.LaplacianEigenMap()
    elif model_name == "modspec":
        model = embcom.ModularitySpectralEmbedding()
    elif model_name == "modspec2":
        model = embcom.ModularitySpectralEmbedding()
    elif model_name == "leigenmap":
        model = embcom.LaplacianEigenMap()
    model.fit(A)
    emb_copy = model.shave_embedding(emb, K + 1)
else:
    emb_copy = emb.copy()
# %%
# Remove nan embedding
results = {}
for dimThreshold in [True, False]:
    for normalize in [True, False]:
        emb = emb_copy.copy()
        if (model_name == "nonbacktracking") & dimThreshold:
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            idx = np.argmax(norm)
            threshold = np.sqrt(norm[idx])
            keep_dims = norm >= threshold
            keep_dims[idx] = False
            if any(keep_dims) is False:
                keep_dims[idx] = True
            emb = emb[:, keep_dims]

        if normalize:
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            emb = np.einsum("ij,j->ij", emb, 1 / np.maximum(norm, 1e-32))

        # Evaluate
        group_ids = KMeans(emb, memberships, metric=metric)

        key = f"normalize~{normalize}_dimThreshold~{dimThreshold}"
        results[key] = group_ids
        #
        print(normalized_mutual_info_score(group_ids, memberships))

# %%
# %%
# Save
#
np.savez(output_file, **results)
