# %%
import glob
import pathlib
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn import metrics
from tqdm import tqdm

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_files"]
    K = snakemake.params["K"]
    output_sim_file = snakemake.output["output_sim_file"]
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/embeddings/two_coms/embeddings/xxx"
    output_file = snakemake.output["output_file"]
    K = 2
    output_sim_file = "unko"


#
# Evaluation
#
def auc_pred_groups(
    emb, group_ids, sample=100000, iterations=5, metric="cosine", subsample=100
):

    clabels, group_ids = np.unique(group_ids, return_inverse=True)
    num_nodes = emb.shape[0]

    def sample_node_pairs():
        n0 = np.random.randint(0, num_nodes, sample)
        n1 = np.random.randint(0, num_nodes, sample)
        g0, g1 = group_ids[n0], group_ids[n1]
        y = np.array(g0 == g1).astype(int)
        return n0, n1, y

    def eval_auc(emb, n1, n0, y, metric):
        e0 = emb[n0, :]
        e1 = emb[n1, :]
        if metric == "cosine":
            s = np.sum(np.multiply(e0, e1), axis=1)
            return metrics.roc_auc_score(y, s), s
        elif metric == "euclidean":
            s = -np.sqrt(np.sum(np.power(e0 - e1, 2), axis=1)).reshape(-1)
            return metrics.roc_auc_score(y, -s), s

    auc_score = []
    sim_vals = []
    is_intra_com_edges = []
    for _ in range(iterations):
        n1, n0, y = sample_node_pairs()
        auc, sval = eval_auc(emb, n1, n0, y, metric)
        auc_score += [auc]
        sim_vals += [sval]
        is_intra_com_edges += [y]
    sim_vals = np.concatenate(sim_vals)
    is_intra_com_edges = np.concatenate(is_intra_com_edges)

    s = np.random.choice(len(sim_vals), subsample)
    sim_vals = sim_vals[s]
    is_intra_com_edges = is_intra_com_edges[s]
    return np.mean(auc_score), sim_vals, is_intra_com_edges


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
emb[np.isnan(emb)] = 0
emb[np.isinf(emb)] = 0
n = int(np.round(emb.shape[0] / K))
group_ids = np.kron(np.arange(K), np.ones(n)).astype(int)

# Evaluate
results = []
dflist = []
for metric in ["cosine", "euclidean"]:
    X = emb.copy()
    X[np.isnan(X)] = 0
    if metric == "cosine":
        X = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
    n = int(X.shape[0] / 2)

    score, sim_vals, is_intra_com_edges = auc_pred_groups(
        X, group_ids, iterations=1, metric=metric
    )

    dh = pd.DataFrame(
        {"score": sim_vals, "is_intra_com_edges": is_intra_com_edges, "metric": metric}
    )
    dflist += [dh]
    results += [{"auc": score, "metric": metric}]

sim_table = pd.concat(dflist)
result_table = pd.DataFrame(results)

# %%
# Save
#
result_table.to_csv(output_file, index=False)
sim_table.to_csv(output_sim_file, index=False)
