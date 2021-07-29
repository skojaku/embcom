"""Plot the AUC of the community dot similarity """
# %%
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/results/two_coms/auc.csv"
    output_file = "../figs/two_coms_auc_n.pdf"

# %%
# Load
#
data_table = pd.read_csv(input_file)
data_table = data_table[~data_table["model"].str.contains("pagerank")]

# %%
# Data filtering
#
n_min = 100
n_max = 10 ** 5
cin_max = 40
metric = "cosine"
model_list = ["node2vec"]
data_table = data_table[data_table.n.between(n_min, n_max)]
data_table = data_table[data_table.cin <= cin_max]
data_table = data_table[data_table.metric == metric]
data_table = data_table[data_table["model"].isin(model_list)]

# %%
#
# Visualization
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
g = sns.FacetGrid(
    data=data_table,
    col_wrap = 4,
    col="dim",
    hue="cin",
    aspect=1,
    height=4,
    palette = "plasma",
    hue_kws={"marker": ["o", "o", "o", "o"]},
)
g.map(sns.lineplot, "n", "auc")

g.set(xscale="log")
g.axes.flat[0].legend(frameon=False)

g.set_ylabels("AUC")
g.set_xlabels("Number of nodes, n")
#sns.lineplot(data = data_table, x = "n", y = "auc", hue = "dim")


# %%
df = data_table.copy()

g = sns.FacetGrid(
    data=df,
    col="cin",
    row="dim",
    hue="model",
    aspect=1,
    height=4,
    hue_kws={"marker": ["o", "s"], "markersize": [10, 10]},
)

g.map(sns.lineplot, "n", "auc")

g.set(xscale="log")
g.axes.flat[0].legend(frameon=False)

g.set_ylabels("AUC")
g.set_xlabels("Number of nodes, n")

g.fig.savefig(output_file, dpi=300, bbox_inches="tight")


# %%
# %%
sns.set_style('white')
sns.set(font_scale=1.5)
sns.set_style('ticks')

fig, axes  = plt.subplots(ncols= 2, figsize=(15,7))

dim = 64
metric = "cosine"
for i, (model, df) in enumerate(data_table.groupby("model")):
    ax = axes[i]
    df = df[df.dim == dim]
    df = df[df.metric == metric]

    ax = sns.lineplot(data = df, x = "n", y="auc", hue = "cin", palette = "coolwarm", marker = "o", markersize = 20, ax = ax)
    ax.set_xlim(90,5e+5 * 1.1)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Number of nodes, n")
    ax.set(xscale="log")
    if i == 0:
        ax.legend(frameon = False)
    else:
        ax.legend().remove()

sns.despine()
#fig.savefig(output_file, dpi=300)
# %%
