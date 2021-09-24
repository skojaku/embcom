""""Calculate the distance between nodes between and within the ground-truth communities"""
import sys
import numpy as np
import pandas as pd
from embcom.PairSampler import PairSampler

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_files"]
    K = int(snakemake.params["K"])
    num_samples = int(snakemake.params["num_samples"])
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/embeddings/multi_coms/embnet_n=1000_K=50_cave=50_cdiff=20_sample=8_model=node2vec_wl=10_dim=50.npz"
    K = 50
    output_file = "tmp.csv"
    num_samples = 10000


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
emb[np.isnan(emb)] = 0
emb[np.isinf(emb)] = 0
n = int(np.round(emb.shape[0] / K))
group_ids = np.kron(np.arange(K), np.ones(n)).astype(int)

# %%
# Evaluate
sampler = PairSampler(group_ids)

pos_pairs = sampler.sample_positive_pairs(num_samples)
neg_pairs = sampler.sample_negative_pairs(num_samples)

pairs = np.vstack(
    [np.hstack([pos_pairs[0], neg_pairs[0]]), np.hstack([pos_pairs[1], neg_pairs[1]])]
).T
isPositive = np.concatenate([np.ones_like(pos_pairs[0]), np.zeros_like(neg_pairs[1])])

# Evaluate the distances
def eval_distance(a, b, metric):
    if metric == "euclidean":
        return np.array(np.linalg.norm(a - b, axis=1)).reshape(-1)
    elif metric == "cosine":
        a = np.einsum("ij,i->ij", a, 1 / np.maximum(1e-30, np.linalg.norm(a, axis=1)))
        b = np.einsum("ij,i->ij", b, 1 / np.maximum(1e-30, np.linalg.norm(b, axis=1)))
        return 1 - np.array(np.sum(a * b, axis=1)).reshape(-1)


# %%
results = []
dflist = []
for metric in ["cosine", "euclidean"]:
    X = emb.copy()
    X[np.isnan(X)] = 0
    d = eval_distance(emb[pairs[:, 0], :], emb[pairs[:, 1], :], metric=metric)
    dh = pd.DataFrame({"distance": d, "inCommunity": isPositive, "metric": metric})
    dflist += [dh]
sim_table = pd.concat(dflist)

# %%
# Save
#
sim_table.to_csv(output_file, index=False)
