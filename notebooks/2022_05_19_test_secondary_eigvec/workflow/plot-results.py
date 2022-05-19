# %%
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import utils

if "snakemake" in sys.modules:
    input_dir = snakemake.input["input_dir"]
    output_file = snakemake.output["output_file"]
else:
    input_dir = "../results"
    output_file = "../data/"

# %5
# Load
#
data_table = utils.load_files(input_dir)


# %%
# Preprocess
#
params = {
    "metric": "euclidean",
    "scoreType": "esim",
    "clustering": "voronoi",
    "normalize": True,
    "multiple_eigenvec": False,
}
plot_data = data_table.copy()

for k, v in params.items():
    if k not in plot_data.columns:
        continue
    plot_data = plot_data[(plot_data[k] == v) | pd.isna(plot_data[k])]

plot_data = utils.to_numeric(
    plot_data, to_int=["cave", "dim", "focal_eigenvec"], to_float=["mu", "score"]
)

# %%
# Plot
sns.set_style("white")
sns.set(font_scale=1.1)
sns.set_style("ticks")
g = sns.FacetGrid(
    data=plot_data, col="cave", row="n", hue="focal_eigenvec", height=3.5,
)

g.map(
    sns.lineplot, "mu", "score", palette="Reds",
)
g.set_xlabels("Mixing rate")
g.set_ylabels("Element-centric similarity")
g.add_legend()

g.fig.savefig("../figs/fig-mixing-vs-performance-focal-eigenvec.pdf")
