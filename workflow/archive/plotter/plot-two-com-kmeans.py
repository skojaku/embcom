"""Plot the AUC of the community dot similarity."""
# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/results/two_coms/res-kmeans.csv"
    output_file = "../figs/two_coms_kmeans_n.pdf"

# %%
# Load
#
data_table = pd.read_csv(input_file)

# %%

# %%
#
# Plot N vs NMI vs dim
#
n_min = 100
n_max = 10 ** 7
cin_max = 40
wl = 10
dim_list = [1, 2, 32, 128]
metric = "cosine"
score_type = "nmi"
model_list = [
    "node2vec",
    "deepwalk",
    "leigenmap",
    "glove",
    "leigenmap",
    "levy-word2vec",
    "modspec",
    "adjspec",
]

df = data_table.copy()

df = df[df.n.between(n_min, n_max)]
df = df[df.dim.isin(dim_list)]
df = df[df.cin <= cin_max]
df = df[df.metric == metric]
df = df[df.wl == wl]
df = df[df["score_type"] == score_type]
df = df[df["model"].isin(model_list)]

# %%
num_cin = len(df["cin"].drop_duplicates().values)

# %% Visualization
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
g = sns.FacetGrid(
    data=df,
    # col_wrap=4,
    row="model",
    col="dim",
    hue="cin",
    aspect=1,
    height=4,
    palette="plasma",
    hue_kws={"marker": ["o" for i in range(num_cin)]},
)
g.map(sns.lineplot, "n", "score")
g.set(xscale="log")
g.axes.flat[1].legend(frameon=False, ncol=7, loc="upper left", bbox_to_anchor=(0, -0.2))
g.set_ylabels("Normalized Mutual Information")
g.set_xlabels("Number of nodes, n")

# %%
#
# N vs NMI vs Window size
#
n_min = 100
n_max = 10 ** 7
cin_max = 40
wl_list = [3, 5, 10]
dim_list = [32]
metric = "cosine"
score_type = "nmi"
model_list = [
    "node2vec",
    "glove",
]

df = data_table.copy()

df = df[df.n.between(n_min, n_max)]
df = df[df.dim.isin(dim_list)]
df = df[df.cin <= cin_max]
df = df[df.metric == metric]
df = df[df.wl.isin(wl_list)]
df = df[df["score_type"] == score_type]
df = df[df["model"].isin(model_list)]

# %% Visualization
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
g = sns.FacetGrid(
    data=df,
    # col_wrap=4,
    row="model",
    col="wl",
    hue="cin",
    aspect=1,
    height=4,
    palette="plasma",
    hue_kws={"marker": ["o" for i in range(num_cin)]},
)
g.map(sns.lineplot, "n", "score")
g.set(xscale="log")
g.axes.flat[4].legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.2))
g.set_ylabels("Normalized Mutual Information")
g.set_xlabels("Number of nodes, n")

# %%
