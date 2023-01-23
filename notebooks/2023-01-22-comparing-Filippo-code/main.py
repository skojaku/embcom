# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-21 17:11:56
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-23 02:08:19
# %%
import filippo_code_base as fb
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/"
    output_file = "../data/"


# ==============
# Parameters
# ==============
n_nodes = 10000  # number of nodes
cave = 5  # Average degree
K = 2  # Number of communities

# ========================
# Load
# ========================
# Get a network generated with the SBM.
from embcom import ModularitySpectralEmbedding, LinearizedNode2Vec

net_list = {}
memberships_list = {}
mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
vec_pred_list = {}
for mu in mu_list:
    node_table_file = f"../../data/multi_partition_model/networks/node_n~{n_nodes}_K~{K}_cave~{cave}_mu~{mu:.2f}_sample~2.npz"
    net_file = f"../../data/multi_partition_model/networks/net_n~{n_nodes}_K~{K}_cave~{cave}_mu~{mu:.2f}_sample~2.npz"

    net = sparse.load_npz(net_file)
    node_table = pd.read_csv(node_table_file)
    memberships = node_table["membership"].values

    # emb = ModularitySpectralEmbedding(p=100, q=40).fit(net).transform(dim=5)
    emb = LinearizedNode2Vec(p=100, q=40).fit(net).transform(dim=5)
    emb = np.real(emb[:, 0]).reshape(-1)

    vec_pred_list[mu] = emb
    net_list[mu] = net
    memberships_list[mu] = memberships


# %% ========================
# Community detection
# ========================
def to_bond(net):
    bond = {}
    n = net.shape[0]
    for i in range(n):
        bond[i] = net.indices[net.indptr[i] : net.indptr[i + 1]].tolist()
    return bond


def dict2vec(vec):
    return np.array(list(vec.values()))


vec_list = {}
cids_list = {}
for mu in tqdm(mu_list):

    net = net_list[mu]

    _, _vec = fb.spectral_modularity(to_bond(net))
    vec = dict2vec(_vec)
    vec_list[mu] = vec

# %% ========================
# Evaluation
# ========================
from sklearn.metrics.cluster import normalized_mutual_info_score


def calc_esim(y, ypred):
    """Element centric similarity."""
    ylab, y = np.unique(y, return_inverse=True)
    ypredlab, ypred = np.unique(ypred, return_inverse=True)

    Ka, Kb = len(ylab), len(ypredlab)

    K = np.maximum(Ka, Kb)
    N = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, K)
    )

    nA = np.array(UA.sum(axis=0)).reshape(-1)
    nB = np.array(UB.sum(axis=0)).reshape(-1)

    nAB = (UA.T @ UB).toarray()
    nAB_rand = np.outer(nA, nB) / N

    # Calc element-centric similarity
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    S = np.sum(np.multiply(Q, (nAB**2))) / N

    # Calc the expected element-centric similarity for random partitions
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :])
    Q = 1 / np.maximum(Q, 1)
    Srand = np.sum(np.multiply(Q, (nAB_rand**2))) / N
    Scorrected = (S - Srand) / (1 - Srand)
    return Scorrected


def get_partition(vec):
    return np.array(vec > 0).astype(int)


results = []
for mu in tqdm(mu_list):
    vec = vec_list[mu]
    cids = get_partition(vec)
    memberships = memberships_list[mu]
    nmi = normalized_mutual_info_score(memberships, cids)
    esim = calc_esim(memberships, cids)
    results.append({"nmi": nmi, "esim": esim, "mu": mu, "model": "Filippo"})

    vec = vec_pred_list[mu]
    cids = get_partition(vec)
    memberships = memberships_list[mu]

    s = ~np.isnan(vec)
    vec = vec[s]
    cids = cids[s]
    memberships = memberships[s]

    nmi = normalized_mutual_info_score(memberships, cids)
    esim = calc_esim(memberships, cids)

    results.append({"nmi": nmi, "esim": esim, "mu": mu, "model": "Sadamori"})

data_table = pd.DataFrame(results)


# %% ========================
# Plot
# ========================

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(7, 5))

ax = sns.lineplot(data=data_table, x="mu", y="esim", hue="model", ax=ax)

ax.axvline(1 - 1 / np.sqrt(cave), ls=":", color="k")

sns.despine()

# %%
# fig.savefig(output_file, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)

# %%
