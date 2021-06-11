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
    input_sim_file = "../data/results/two_coms/sim_vals.csv"
    output_file = "../figs/two_coms_auc_n.pdf"

# %%
# Load
#
data_table = pd.read_csv(input_file)
simval_table = pd.read_csv(input_sim_file)

# %%
# Preprocess
#
nmin = 100
data_table = data_table[data_table.n >= nmin]
simval_table = simval_table[simval_table.n >= nmin]

# %%
#
# Visualize the distribution of matrix elements
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

df = simval_table.copy()
g = sns.FacetGrid(
    data=df,
    row="metric",
    col="model",
    hue="is_intra_com_edges",
    aspect=1.5,
    sharey=False,
)

g.map(sns.lineplot, "n", "score", ci="sd")
g.set(xscale="log")
g.axes.flat[0].legend(frameon=False)
g.set_ylabels("Distance")
g.set_xlabels("Number of nodes, n")


# %%
# %%
df = data_table.copy()
g = sns.FacetGrid(
    data=df,
    col="metric",
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

#
# Save
#

# %%
data_table

# %%
