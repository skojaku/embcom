# %%
import sys

import emlens
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/derived/sim_R/rvals.csv"
    output_file = "../figs/rvals.pdf"

# %%
# Load
#
res_table = pd.read_csv(input_file)


# %%
# Preprocess
#
df = res_table.copy()

# Filtering
df = df[df["T"] <= 20]

#  Compute variance
dh = df.copy()
dh = dh[dh["rtype"].apply(lambda x: "sim" not in x)]
dh["rvar"] = dh["error"]
dh = dh[["rvar", "n", "T", "rtype"]]

dg = df.copy()
dg = dg[dg["rtype"].apply(lambda x: "sim" in x)]
dg = (
    dg.groupby(["n", "T", "rtype"])
    .agg("var")["r"]
    .reset_index()
    .rename(columns={"r": "rvar"})
)[["rvar", "n", "T", "rtype"]]
dg = pd.concat([dh, dg])
df = pd.merge(df, dg, on=["n", "T", "rtype"])

# Format
df1 = df[["r", "T", "rtype", "n"]]
df1["stat"] = "mean"
df2 = df[["rvar", "T", "rtype", "n"]].rename(columns={"rvar": "r"})
df2["stat"] = "var"
df = pd.concat([df1, df2])

# Rename
rtype_name = {
    "in": "Win",
    "out": "Wout",
    "sim-in": "Win (sim)",
    "sim-out": "Wout (sim)",
}
df["rtype"] = df["rtype"].map(rtype_name)

# %%


# %%
# Visualization
#
sns.set_style("white")
sns.set(font_scale=1.1)
sns.set_style("ticks")

cmap = sns.color_palette().as_hex()
cmap = [
    cmap[0],
    cmap[3],
    sns.light_palette(cmap[0], n_colors=5)[2],
    sns.light_palette(cmap[3], n_colors=5)[2],
]
g = sns.FacetGrid(
    data=df,
    col="T",
    row="stat",
    hue="rtype",
    hue_order=rtype_name.values(),
    hue_kws={
        "marker": ["s", "s", "o", "o"],
        "ls": [":", ":", "-", "-"],
        "color": cmap,
        "markerfacecolor": cmap,
        "markeredgecolor": ["k", "k", "k", "k"],
    },
    sharey=False,
)

g = g.map(sns.lineplot, "n", "r")
g.set_ylabels("$W_{ij}$")
g.set_xlabels("")
g.fig.text(0.5, 0.01, "Number of nodes, $n$")
g.axes[0, 0].set_ylabel("$E[W_{ij}]$")
g.axes[1, 0].set_ylabel("$(W_{ij} - E[W_{ij}])^2$")
g.set_titles("")

subcap = "ABCDEFGHIJK"
for i, ax in enumerate(g.axes.flat):
    ax.annotate(
        r"%s" % (subcap[i]), (-0.15, 1.05), xycoords="axes fraction",
    )
Tlist = df["T"].drop_duplicates().sort_values().values
for i in range(4):
    g.axes[0, i].set_title("T={}".format(Tlist[i]))

g.axes[0, 0].legend(frameon=False)
g.despine()

#
# Save
#
g.fig.savefig(output_file, dpi=300, bbox_inches="tight")
