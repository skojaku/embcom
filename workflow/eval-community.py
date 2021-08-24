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
    emb_files = snakemake.input["emb_files"]
    output_file = snakemake.output["output_file"]
    output_sim_file = snakemake.output["output_sim_file"]
else:
    emb_files = list(glob.glob("../data/embeddings/two_coms/embeddings/*"))
    output_file = "../data/results/two_coms/auc.csv"
    output_sim_file = "../data/results/two_coms/sim_vals.csv"

#
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
# G
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


def eval_clu(df):

    # Load emebdding
    emb_list = {}
    for _i, row in df.iterrows():
        emb = np.load(row["emb_file"])["emb"]
        emb = emb.copy(order="C").astype(np.float32)
        emb_list[row["emb_file"]] = emb

    # Evaluate
    results = []
    dflist = []
    for metric in ["cosine", "euclidean"]:
        for _i, row in df.copy().iterrows():
            emb = emb_list[row["emb_file"]]
            X = emb.copy()
            X[np.isnan(X)] = 0
            if metric == "cosine":
                X = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
            n = int(X.shape[0] / 2)
            y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
            score, sim_vals, is_intra_com_edges = auc_pred_groups(
                X, y, iterations=1, metric=metric
            )

            dh = pd.DataFrame(
                {
                    "score": sim_vals,
                    "is_intra_com_edges": is_intra_com_edges,
                    "metric": metric,
                }
            )
            for k, v in row.items():
                dh[k] = v

            row["auc"] = score
            row["metric"] = metric

            dflist += [dh]
            results += [row]
    dh = pd.concat(dflist)
    return (results, dh)


list_results = Parallel(n_jobs=3)(
    delayed(eval_clu)(df) for emb_file, df in tqdm(emb_file_table.groupby("emb_file"))
)

# %%
# Preprocess
#
results = []
for res in list_results:
    results += res[0]
result_table = pd.DataFrame(results)

results = []
for res in list_results:
    results += [res[1]]
sim_table = pd.concat(results)

# %%
# Save
#
result_table.to_csv(output_file, index=False)
sim_table.to_csv(output_sim_file, index=False)

#
# Save
#

# %%
