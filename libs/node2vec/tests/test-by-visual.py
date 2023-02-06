# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:41:52
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-19 22:23:52
# %%
import node2vecs

node2vecs.__path__
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
# %%
# ======================
# Karate club
# ======================

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]

# %%
A.data

# %%
n_nodes = A.shape[0]
dim = 32
# model = node2vecs.TorchNode2Vec(
#    num_walks=100, batch_size=1024, negative=1, epochs=10, device="cuda:1"
# )
# model = node2vecs.TorchLaplacianEigenMap(num_walks=50, negative=1)
model = node2vecs.TorchModularity(num_walks=50, negative=1)
model.fit(A)
emb = model.transform()
# model = node2vecs.GensimNode2Vec()
# model.fit(A)
# emb = model.transform()

# %%

# nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
xy = PCA(n_components=2).fit_transform(emb)
clf = LinearDiscriminantAnalysis(n_components=1)
x = clf.fit_transform(emb, labels)
# xy[:, 0] = x.reshape(-1)
plot_data = pd.DataFrame(
    {"x": xy[:, 0], "y": xy[:, 1], "model": "non-linear", "label": labels}
)
# %%

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
# fig, ax = plt.subplots(figsize=(7,5))

g = sns.FacetGrid(data=plot_data, col="model")
g.map(sns.scatterplot, "x", "y", "label")

# fig.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
# %%
clf = LinearDiscriminantAnalysis(n_components=1)
xy = clf.fit_transform(emb, labels)
# xy = clf.fit_transform(emb_linear, labels)
clf.score(emb, labels)
# %%
sns.heatmap(emb @ emb.T, cmap="coolwarm", center=0)

# %%
emb @ emb.T
# %%
# ===================
# Airport network
# ===================
# Node attributes
node_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/node-table-airport.csv"
)

# Edge table
edge_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/edge-table-airport.csv"
)
# net = nx.adjacency_matrix(nx.from_pandas_edgelist(edge_table))

net = sparse.csr_matrix(
    (
        edge_table["weight"].values,
        (edge_table["source"].values, edge_table["target"].values),
    ),
    shape=(node_table.shape[0], node_table.shape[0]),
)

s = ~pd.isna(node_table["region"])
labels = node_table[s]["region"].values
net = net[s, :][:, s]
net.data = net.data * 0 + 1
# %%
model = node2vecs.TorchModularity(
    num_walks=25, negative=1, num_workers=5, device="cuda:1"
)
model.fit(net)
emb = model.transform()

# %%
clf = LinearDiscriminantAnalysis(n_components=2)
xy = clf.fit_transform(emb, labels)
plot_data = pd.DataFrame(
    {"x": xy[:, 0], "y": xy[:, 1], "model": "non-linear", "label": labels}
)

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

sns.scatterplot(data = plot_data, x = "x", y = "y", hue = "label")
print(clf.score(emb, labels))

# %%
emb @ emb.T / np.sqrt(net.sum())