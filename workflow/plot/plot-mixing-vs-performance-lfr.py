# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-07-11 22:08:07
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-28 14:38:54
# %%
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import color_palette as cp
import textwrap

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    model_names = snakemake.params["model_names"]
    params["name"] = model_names
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
else:
    input_file = "../../data/lfr/evaluations/all-result.csv"
    output_file = "../data/"

    params = {
        "dim": 64,
        "n": 100000,
        "metric": "cosine",
        "length": 10,
        "k": 50,
        "clustering": "kmeans",
        "score_type": "esim",
        "dimThreshold": True,
        # "normalize": True,
    }

#
# Load
#
data_table = pd.read_csv(input_file)

#
plot_data = data_table.copy()
for k, v in params.items():
    if k not in plot_data.columns:
        continue
    if not isinstance(v, list):
        v = [v]
    plot_data = plot_data[(plot_data[k].isin(v)) | pd.isna(plot_data[k])]

plot_data = plot_data[plot_data["name"] != "levy-word2vec"]

# %%
plot_data

# %%
#
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")

model_list = cp.get_model_order()
data_model_list = plot_data["name"].unique().tolist()
model_list = [k for k in model_list if k in data_model_list]

model_color = cp.get_model_colors()

c1 = sns.color_palette("Set2")
c2 = sns.color_palette("Set1")
c3 = sns.color_palette(desat=0.6)

model_markers = cp.get_model_markers()
model_linestyles = cp.get_model_linestyles()
model_names = cp.get_model_names()

model_markers = {k: model_markers[k] for k in model_list}
model_linestyles = [model_linestyles[k] for k in model_list]
model_names = {k: model_names[k] for k in model_list}

fig, ax = plt.subplots(figsize=(5.5, 5.5))

ax = sns.lineplot(
    data=plot_data,
    x="mu",
    y="score",
    hue="name",
    style="name",
    markers=model_markers,
    style_order=model_list,
    hue_order=model_list,
    palette=model_color,
    markeredgecolor="k",
    ax=ax,
)

ax.set_xlabel(r"Mixing rate, $\mu$")

if params["score_type"] == "nmi":
    ax.set_ylabel(r"Normalized Mutual Information")
elif params["score_type"] == "esim":
    ax.set_ylabel(r"Element-centric similarity")

ax.set_ylim(-0.03, 1.05)
# mu_max = 1 - 1 / np.sqrt(params["cave"])
# ax.axvline(mu_max, color="black", linestyle="--")

current_handles, current_labels = ax.get_legend_handles_labels()
new_labels = [model_names[l] for l in current_labels]
lgd = ax.legend(
    current_handles,
    new_labels,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(-0.15, -0.15),
    ncol=3,
    fontsize=7,
    # labels=[model_names[m] for m in model_list],
)
sns.despine()

if title is not None:
    ax.set_title(textwrap.fill(title, width=42))

fig.savefig(
    output_file,
    bbox_extra_artists=(lgd,),
    bbox_inches="tight",
    dpi=300,
)
# %%
