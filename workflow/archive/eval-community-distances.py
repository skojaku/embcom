""""Calculate the distance between nodes between and within the ground-truth
communities."""
# %%
import sys

import numpy as np
import pandas as pd

from embcom.PairSampler import PairSampler

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = (
        snakemake.input["com_file"] if "com_file" in snakemake.input.keys() else None
    )
    K = int(snakemake.params["K"])
    num_samples = int(snakemake.params["num_samples"])
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/embeddings/lfr/embnet_n=1000_k=10_maxk=100_minc=20_maxc=100_tau=2_tau2=1_mu=0.25_sample=4_model=glove_wl=10_dim=64.npz"
    com_file = "../data/networks/lfr/community_n=1000_k=10_maxk=100_minc=20_maxc=100_tau=2_tau2=1_mu=0.25_sample=4.npz"

    K = 50
    output_file = "tmp.csv"
    num_samples = 10000

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

# %%
# Evaluate
sampler = PairSampler(group_ids)

anc, pos, neg = sampler.sample_anchor_positive_negative_triplet(num_samples)

# %%
# Evaluate the distances
def eval_distance(a, b, metric):
    if metric == "euclidean":
        return np.array(np.linalg.norm(a - b, axis=1)).reshape(-1)
    elif metric == "cosine":
        a = np.einsum("ij,i->ij", a, 1 / np.maximum(1e-30, np.linalg.norm(a, axis=1)))
        b = np.einsum("ij,i->ij", b, 1 / np.maximum(1e-30, np.linalg.norm(b, axis=1)))
        return 1 - np.array(np.sum(a * b, axis=1)).reshape(-1)


results = []
dflist = []
for metric in ["cosine", "euclidean"]:
    X = emb.copy()
    X[np.isnan(X)] = 0
    dpos = eval_distance(emb[anc, :], emb[pos, :], metric=metric)
    dneg = eval_distance(emb[anc, :], emb[neg, :], metric=metric)
    score = np.log(np.maximum(1e-12, dneg)) - np.log(np.maximum(1e-12, dpos))
    dh = pd.DataFrame({"score": score, "metric": metric})
    dflist += [dh]
sim_table = pd.concat(dflist)
sim_table = sim_table.groupby(["metric"]).mean().reset_index()

# %%
# Save
#
sim_table.to_csv(output_file, index=False)
