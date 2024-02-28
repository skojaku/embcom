"""Evaluate the detected communities using the element-centric similarity."""

# %%
import sys
import numpy as np
import pandas as pd

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    model_name = params["model_name"]
    n_neighbors = 100

else:
    emb_file = "../../data/multi_partition_model/embedding/n~1000_K~2_cave~50_mu~0.70_sample~1_model_name~leigenmap_window_length~10_dim~64.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~2500_K~2_cave~50_mu~0.70_sample~1.npz"
    model_name = "leigenmap"
    output_file = "unko"
    metric = "cosine"
    n_neighbors = 100


from sklearn import cluster
import faiss
import igraph


def ModularityClustering(emb, group_ids, n_neighbors, metric="euclidean"):
    K = np.max(group_ids) + 1
    if metric == "cosine":
        X = np.einsum(
            "ij,i->ij", emb, 1 / np.maximum(np.linalg.norm(emb, axis=1), 1e-24)
        )
        X = emb

    # Convert data to float32 for faiss
    emb_faiss = X.astype("float32")

    # Create a faiss index (Flat index for brute-force search)
    index = faiss.IndexFlatIP(emb_faiss.shape[1])

    # Add vectors to the index
    index.add(emb_faiss)

    # Search the index
    _, I = index.search(
        emb_faiss, n_neighbors + 1
    )  # Search for k+1 to include the point itself

    # Construct a k-Nearest Neighbors graph
    # Exclude the first column since it's the point itself
    I = I[:, 1:]

    # Convert indices matrix to a sparse matrix representation of the graph
    src = np.repeat(np.arange(emb_faiss.shape[0]), n_neighbors)
    trg = I.flatten()
    edge_list = tuple(zip(src, trg))
    g = igraph.Graph(edge_list, directed=False)
    partition = g.community_leiden(objective_function="modularity")

    # Extract membership information
    community_membership = partition.membership
    return community_membership


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# Remove nan embedding
emb = np.nan_to_num(emb)

emb_copy = emb.copy()

# Normalize the eigenvector by dimensions
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
        group_ids = ModularityClustering(
            emb, memberships, n_neighbors=n_neighbors, metric=metric
        )

        key = f"normalize~{normalize}_dimThreshold~{dimThreshold}"
        results[key] = group_ids

# %%
# Save
#
np.savez(output_file, **results)
