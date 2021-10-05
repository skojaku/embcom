""""Calculate the distance between nodes between and within the ground-truth
communities."""
# %%
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = (
        snakemake.input["com_file"] if "com_file" in snakemake.input.keys() else None
    )
    K = int(snakemake.params["K"])
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/embeddings/multi_coms/embnet_n=1000_K=50_cave=50_cdiff=20_sample=8_model=node2vec_wl=10_dim=50.npz"
    K = 50
    output_file = "tmp.csv"


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
clf = LinearDiscriminantAnalysis()

scores = np.zeros(K)
for k in range(K):
    y = (group_ids == k).astype(int)

    if emb.shape[1] >= 2:
        clf.fit(emb, y)
        x = clf.transform(emb)
        x = np.array(x).reshape(-1)
    else:
        x = np.array(emb.copy()).reshape(-1)
    scores[k] = roc_auc_score(y, x)
score = np.mean(scores)

pd.DataFrame([{"score": score}]).to_csv(output_file, index=False)
