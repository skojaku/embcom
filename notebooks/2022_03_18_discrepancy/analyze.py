"""Investigate the discrepancy between the actual and analytical spectrum density"""
# %%
from matplotlib.pyplot import plot
import numpy as np
from scipy import sparse
import utils
from scipy import stats
import pandas as pd
import sys
import seaborn as sns
import utils

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
else:
    input_files = "data/results"
    output_file = "../data/"

# Load
data_table = utils.load_files(
    input_files,
    dtypes={
        "Cave": "int",
        "rate": "float",
        "score_node2vec": "float",
        "score_node2vec_lim": "float",
        "pearson": "float",
        "L": int,
        "N": int,
    },
)


# %% Plot the correlation between the largest eigenvector
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

plot_data = data_table.copy()
g = sns.FacetGrid(
    data=plot_data,
    row="Cave",
    col="N",
    hue="L",
    hue_order=[5, 10],
    height=3,
    aspect=1.2,
)
g.map(sns.lineplot, "rate", "pearson", label="pearson", marker="o")
g.axes[0, 0].legend(frameon=False).set_title("L")
sns.despine()

# %%
df = data_table[
    ["filename", "Cave", "rate", "N", "L", "score_node2vec", "score_node2vec_lim"]
]
df = data_table.drop(columns=["score_node2vec_lim"]).rename(
    columns={"score_node2vec": "score"}
)
df["model"] = "original node2vec"
dg = data_table.drop(columns=["score_node2vec"]).rename(
    columns={"score_node2vec_lim": "score"}
)
dg["model"] = "linearized node2vec"
plot_data = pd.concat([df, dg])
# %% Mixing rate vs score: in case of large degree
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
import matplotlib.pyplot as plt

_plot_data = plot_data.copy()
_plot_data = plot_data[plot_data["N"] == 5000]
_plot_data = _plot_data[_plot_data["L"] == 10]
g = sns.FacetGrid(
    data=_plot_data,
    row="Cave",
    col="N",
    hue="model",
    hue_order=["original node2vec", "linearized node2vec"],
    height=3.5,
    aspect=1.5,
)

g.map(sns.lineplot, "rate", "score", marker="o")
g.axes[0, 0].legend(frameon=False, loc="lower left")
g.axes[0, 0].axvline(1 - 1 / np.sqrt(10), color="k", linestyle="--")
g.axes[2, 0].legend(frameon=False, loc="lower left")
g.axes[2, 0].axvline(1 - 1 / np.sqrt(50), color="k", linestyle="--")

sns.despine()
# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
import matplotlib.pyplot as plt

_plot_data = plot_data.copy()
_plot_data = plot_data[plot_data["N"] == 1000]
_plot_data = _plot_data[_plot_data["model"] == "linearized node2vec"]
# _plot_data = _plot_data[_plot_data["L"] == 10]
g = sns.FacetGrid(
    data=_plot_data,
    row="Cave",
    col="N",
    hue="L",
    # hue_order=["linearized node2vec"],
    height=3.5,
    aspect=1.5,
)

g.map(sns.lineplot, "rate", "score", marker="o")
g.axes[2, 0].legend(frameon=False, loc="lower left")
g.axes[2, 0].axvline(1 - 1 / np.sqrt(50), color="k", linestyle="--")

sns.despine()
