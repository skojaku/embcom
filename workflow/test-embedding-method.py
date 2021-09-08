# %%
%load_ext autoreload
%autoreload 2

import logging
import os
import sys

print(sys.path)
#import emlens
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import rc
from scipy import sparse, stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import embcom

#
# Load the test data
#
net = sparse.load_npz("../data/test/net.npz")
node_table = pd.read_csv("../data/test/node.csv", sep="\t")

community = node_table["community"]
is_hub_nodes = node_table["is_hub_nodes"]

net = net + net.T
# %%
models = {}

# %%
metric = "euclidean"
dim = 32
window_length = 10
models["node2vec"] = embcom.embeddings.Node2Vec(window_length = window_length)
models["deepwalk"] = embcom.embeddings.DeepWalk(window_length = window_length)
models["adjspec"] = embcom.embeddings.AdjacencySpectralEmbedding()
models["modspec"] = embcom.embeddings.ModularitySpectralEmbedding()
models["leigen"] = embcom.embeddings.LaplacianEigenMap()
models["glove"] = embcom.embeddings.Glove()

# %%
embs = {}
for name, model in models.items():
    model = model.fit(net)
    emb = model.transform(dim)
    if metric == "cosine":
        emb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
    embs[name] = emb
    models[name] = model

# %%

xys = {}
for name, emb in embs.items():
    xys[name] = PCA(n_components=2).fit_transform(emb)

dflist = []
for i, (name, xy) in enumerate(xys.items()):
    df = pd.DataFrame({"group": community})
    df["deg"] = np.array(net.sum(axis=1)).reshape(-1)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    df["model"] = name
    dflist += [df]
df = pd.concat(dflist)

g = sns.FacetGrid(
    data=df, col="model", hue="group", col_wrap=2, height=6, sharex=False, sharey=False
)


for i, (model, dg) in enumerate(df.groupby("model")):
    ax = g.axes.flat[i]
    sns.scatterplot(
        data=dg,
        x="x",
        y="y",
        size="deg",
        sizes=(1, 200),
        hue="group",
        edgecolor="black",
        linewidth=0.2,
        ax=ax,
    )
    ax.set_title(model)
    if i == 0:
        ax.legend(frameon=False)
    else:
        ax.legend().remove()

g.set_xlabels("")
g.set_ylabels("")
sns.despine()
g.axes[0].legend(frameon=False)

# %%

for name, emb in embs.items():
    score = embcom.metrics.eval_separation_lda(emb, community)
    print("{method}: {score}".format(method = name, score = score))
    #xys[name] = PCA(n_components=2).fit_transform(emb)
# %%
