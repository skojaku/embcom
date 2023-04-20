# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:41:52
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-03 17:38:42
# %%
import numpy as np
from scipy import sparse


def fastRP(net, dim, window_size, beta=-1, s=3.0):

    n_nodes = net.shape[0]

    outdeg = np.array(net.sum(axis=1)).reshape(-1)
    indeg = np.array(net.sum(axis=0)).reshape(-1)

    P = sparse.diags(1 / np.maximum(1, outdeg)) @ net  # Transition matrix
    L = sparse.diags(np.power(indeg.astype(float), beta))
    X = sparse.random(
        n_nodes,
        dim,
        density=1 / s,
        data_rvs=lambda x: np.sqrt(s) * (2 * np.random.randint(2, size=x) - 1),
    ).toarray()
    X0 = (P @ L) @ X.copy()  # to include the self-loops
    h = np.ones((n_nodes, 1))
    h0 = h.copy()
    for _ in range(window_size):
        R = P @ X + X0
        h = P @ h + h0
    X = sparse.diags(1.0 / np.maximum(np.array(h).reshape(-1), 1e-8)) @ X

    return R


import node2vecs

node2vecs.__path__
# %%
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]

n_nodes = A.shape[0]
dim = 32
model = node2vecs.TorchNode2Vec(
    num_walks=100, batch_size=1024, negative=1, epochs=10, device="cuda:1"
)
model = node2vecs.TorchLaplacianEigenMap(num_walks=50, negative=1)
# model = node2vecs.TorchModularity(num_walks=50, negative=1)
model.fit(A)
emb = model.transform()
# model = node2vecs.GensimNode2Vec()
# model.fit(A)
# emb = model.transform()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.decomposition import PCA

# nemb = np.einsum("ij,i->ij", emb, 1 / np.linalg.norm(emb, axis=1))
xy = PCA(n_components=2).fit_transform(emb)
clf = LinearDiscriminantAnalysis(n_components=1)
x = clf.fit_transform(emb, labels)
# xy[:, 0] = x.reshape(-1)
plot_data = pd.DataFrame(
    {"x": xy[:, 0], "y": xy[:, 1], "model": "non-linear", "label": labels}
)
# %%
import seaborn as sns
import matplotlib.pyplot as plt

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
