# %%
import glob
import sys

import pandas as pd
import utils

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
else:
    input_files = list(glob.glob("../data/results/two_coms/auc/*"))
    output_file = "../data/"

# input_files = list(glob.glob(input_dir + "/*"))

#
# Load
#
df = utils.get_params(input_files)

# %%
# Preprocess
#
filenames = df["filename"].drop_duplicates().values
dglist = []
for filename in filenames:
    dg = pd.read_csv(filename)
    dg["filename"] = filename
    dglist += [dg]
dg = pd.concat(dglist)
df = pd.merge(df, dg, on="filename")

# %%
# Save
#
df.to_csv(output_file, index=False)
