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
    input_file = "../data/lfr/results/all-result.csv"
    output_file = "../data/"

    params = {
        "dim": 64,
        "n": 100000,
        "metric": "cosine",
        "length": 10,
        "k": 50,
        "score_type": "esim",
        "clustering": [
            "louvain",
            "clustering~kmeans_normalize~True",
            "clustering~voronoi_normalize~True",
        ],
    }

#
# Load
#
data_table = pd.read_csv(input_file)

# %%

#
plot_data = data_table.copy()
for k, v in params.items():
    if k not in plot_data.columns:
        continue
    if not isinstance(v, list):
        v = [v]
    plot_data = plot_data[(plot_data[k].isin(v)) | pd.isna(plot_data[k])]

plot_data = plot_data[plot_data["name"] != "levy-word2vec"]
plot_data
# %%
#
# Plot
#
sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")

model_list = cp.get_model_order()
model_color = cp.get_model_colors()
model_markers = cp.get_model_markers()
model_linestyles = list(cp.get_model_linestyles().values())
model_names = cp.get_model_names()

fig, ax = plt.subplots(figsize=(7, 5))

ax = sns.lineplot(
    data=plot_data,
    x="mu",
    y="score",
    hue="clustering",
    # markers=model_markers,
    # style_order=model_list,
    # hue_order=model_list,
    # palette=model_color,
    markeredgecolor="k",
    ax=ax,
)

ax.set_xlabel(r"Mixing rate, $\mu$")

if params["score_type"] == "nmi":
    ax.set_ylabel(r"Normalized Mutual Information")
elif params["score_type"] == "esim":
    ax.set_ylabel(r"Element-centric similarity")

ax.set_ylim(-0.03, 1.05)
ax.legend(bbox_to_anchor = (0.5, -0.15), loc = "upper center", frameon = False, ncol = 2)
# mu_max = 1 - 1 / np.sqrt(params["cave"])
# ax.axvline(mu_max, color="black", linestyle="--")

sns.despine()

# fig.savefig(
#    output_file,
#    bbox_extra_artists=(lgd,),
#    bbox_inches="tight",
#    dpi=300,
# )

# %%
plot_data

# %%
