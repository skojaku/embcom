# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    filter_params = snakemake.params
    col_filters = {k: filter_params[k] for k in filter_params.keys()}
else:
    input_file = "../../data/results/multi_coms/results/separatability.csv"
    output_file = "distances.pdf"

    col_filters = {
        "K": 2,
        "metric": "euclidean",
        "wl": 10,
        "dim": 64,
        "cave": 50,
        "cdiff": 160,
    }


def data_filtering(data_table, col_filters, overwrite=None):

    if overwrite:
        for k, v in overwrite.items():
            col_filters[k] = v

    plot_data = data_table.copy()
    for col, v in col_filters.items():
        if col not in plot_data.columns:
            continue
        if isinstance(v, list):
            s = plot_data[col].isin(v)
        else:
            s = plot_data[col] == v

        plot_data = plot_data[s | pd.isna(plot_data[col])]
    return plot_data


#
# Load
#
data_table = pd.read_csv(input_file)
data_table = data_table.rename(columns={"distance": "score"})


# %%
# Preprocess
#
plot_data = data_filtering(data_table.copy(), col_filters=col_filters)

# %%

method_labels = {
    "node2vec": "node2vec",
    "glove": "Glove",
    "leigenmap": "L-EigenMap",
    "modspec": "Modularity",
    "nonbacktracking": "Non-backtracking",
}

plot_data["Model"] = plot_data["model"].map(method_labels)
plot_data["score"] = 2 * np.abs(
    plot_data["score"] - 0.5
)  # shift and rescale to the score to vary in [0,1].

#
# Styles
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

cmap = sns.color_palette(desat=0.8).as_hex()

model2color = {
    "node2vec": cmap[0],
    "glove": cmap[1],
    "leigenmap": cmap[2],
    "modspec": cmap[3],
    "nonbacktracking": cmap[4],
}
model2markeredgecolor = {
    "node2vec": "k",
    "glove": "k",
    "leigenmap": "#8d8d8d",
    "modspec": "#8d8d8d",
    "nonbacktracking": "k",
}
model2marker = {
    "node2vec": "s",
    "glove": "o",
    "leigenmap": ">",
    "modspec": "^",
    "nonbacktracking": "D",
}
model2ls = {
    "node2vec": "",
    "glove": (5, 1),
    "leigenmap": (3, 1),
    "modspec": (1, 1),
    "nonbacktracking": (1, 1),
}

# hue order. Flip the order for legend
hue_order = list(method_labels.values())
palette = {v: model2color[k] for k, v in method_labels.items()}
markers = {v: model2marker[k] for k, v in method_labels.items()}  # [::-1]
markeredgecolors = [model2markeredgecolor[i] for i in method_labels.keys()][::-1]
linestyles = {v: model2ls[k] for k, v in method_labels.items()}

# hue parameter kwd GridFacet
hue_kws = {
    "hue_order": hue_order,
    "markers": markers,
    "markersize": 10,
    "dashes": linestyles,
    # "markersize": 10,  # [6 for i in range(len(palette))],
    # "markerfacecolor": palette,
    # "color": palette,
    # "markeredgecolor": markeredgecolors,
}

#
# Plot
#
fig, ax = plt.subplots(figsize=(5, 5))

ax = sns.lineplot(
    data=plot_data,
    x="n",
    y="score",
    palette=palette,
    style="Model",
    hue="Model",
    ax=ax,
    **hue_kws
)
ax.legend(
    frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.01, 0.01), fontsize=12,
).set_title("Model")
ax.legend().remove()
ax.set_xscale("log")
ax.set_xlabel(r"")
ax.set_ylabel(r"")
# ax.set_xlabel(r"Number of nodes, $n$")
# ax.set_ylabel(r"Separatability")
sns.despine()
plt.tight_layout()
fig.savefig(output_file, dpi=300, bbox_inches="tight")
