# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:41:52
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-14 14:51:11
# %%
import node2vecs
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]

n_nodes = A.shape[0]
dim = 32

sampler = node2vecs.RandomWalkSampler(A, walk_length=80)

# Word2Vec model
model = node2vecs.Word2Vec(n_nodes=n_nodes, dim=dim, orthogonal_constraint=False)

# Set up negative sampler
dataset = node2vecs.NegativeSamplingDataset(
    seqs=sampler, window=10, epochs=3, context_window_type="double"
)

# Set up the loss function
# loss_func = node2vecs.ModularityTripletLoss(model=model)
loss_func = node2vecs.TripletLoss(model)

# Train
node2vecs.train(
    model=model,
    dataset=dataset,
    loss_func=loss_func,
    batch_size=1000,
    device="cpu",
    num_workers=5,
)
emb = model.embedding()

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
xy[:, 0] = x.reshape(-1)
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

emb.T @ emb

# %%
