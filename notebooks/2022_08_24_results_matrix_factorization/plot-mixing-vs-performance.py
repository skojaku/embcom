# %%
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import color_palette as cp

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
else:
    input_file = "../../data/multi_partition_model/all-result.csv"
    output_file = "results_matrix_factorization.pdf"

    params = {
        "q": [2],
        "dim": 64,
        "n": 100000,
        "metric": "cosine",
        "length": 10,
        "clustering": "voronoi",
        "score_type": "esim",
        "cave": [50, 10],
        "name": ["leigenmap", "modspec", "nonbacktracking", "linearized-node2vec"],
        "dimThreshold": False,
        "normalize": True,
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
plot_data["name"].unique()
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

muted_colors = sns.color_palette().as_hex()
colors = sns.color_palette("muted").as_hex()
model_color = {
    "nonbacktracking": "#2d2d2d",
    "leigenmap": muted_colors[3],
    "modspec": muted_colors[0],
    "linearized-node2vec": muted_colors[1],
}

model_markers = cp.get_model_markers()
model_linestyles = cp.get_model_linestyles()
model_names = cp.get_model_names()

model_markers = {k: model_markers[k] for k in model_list}
model_linestyles = {k: model_linestyles[k] for k in model_list}
model_names = {k: model_names[k] for k in model_list}

fig, axes = plt.subplots(figsize=(11, 5), ncols=2)

ax = sns.lineplot(
    data=plot_data[plot_data["cave"] == 50],
    x="mu",
    y="score",
    hue="name",
    style="name",
    # dashes=model_linestyles,
    markers=model_markers,
    style_order=model_list,
    hue_order=model_list,
    palette=model_color,
    markeredgecolor="k",
    ax=axes[0],
)

ax.set_xlabel("")
if params["score_type"] == "nmi":
    ax.set_ylabel(r"Normalized Mutual Information")
else:
    ax.set_ylabel(r"Element-centric similarity")

mu_max = 1 - 1 / np.sqrt(params["cave"][0])
ax.axvline(mu_max, color="black", linestyle="--")
ax.set_title(
    r"$c_{{\rm ave}} = {cave}$".format(cave=params["cave"][0]),
    fontsize=20,
    va="center",
    ha="center",
)

current_handles, current_labels = ax.get_legend_handles_labels()
new_labels = [model_names[l] for l in current_labels]
lgd = ax.legend(
    current_handles,
    new_labels,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.2, -0.2),
    ncol=2,
    fontsize=12,
)
ax.set_ylim(-0.01)
ax.annotate(
    "A",
    xy=(0.05, 0.84),
    weight="bold",
    fontsize=22,
    xycoords="axes fraction",
)

#
ax = sns.lineplot(
    data=plot_data[plot_data["cave"] == 10],
    x="mu",
    y="score",
    hue="name",
    style="name",
    markers=model_markers,
    style_order=model_list,
    hue_order=model_list,
    palette=model_color,
    markeredgecolor="k",
    ax=axes[1],
)
mu_max = 1 - 1 / np.sqrt(params["cave"][1])
ax.axvline(mu_max, color="black", linestyle="--")
ax.set_ylim(-0.01)
ax.legend().remove()
ax.set_ylabel("")
ax.set_xlabel("")
ax.annotate(
    "B",
    xy=(0.05, 0.84),
    weight="bold",
    fontsize=22,
    xycoords="axes fraction",
)
ax.set_title(
    r"$c_{{\rm ave}} = {cave}$".format(cave=params["cave"][1]),
    fontsize=20,
    va="center",
    ha="center",
)

fig.text(0.5, 0.03, r"Mixing rate, $\mu$", ha="center", va="top")
sns.despine()

fig.savefig(
    output_file,
    bbox_extra_artists=(lgd,),
    bbox_inches="tight",
    dpi=300,
)

# %%
