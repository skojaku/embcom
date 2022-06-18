"""Putting together results into a data table"""
import sys
import glob
import glob
import pathlib
import sys
import pandas as pd
from tqdm import tqdm


def load_files(dirname):
    if isinstance(dirname, str):
        input_files = list(glob.glob(dirname + "/*"))
    else:
        input_files = dirname

    def get_params(filenames):
        def _get_params(filename, sep="~"):
            params = pathlib.Path(filename).stem.split("_")
            retval = {"filename": filename}
            for p in params:
                if sep not in p:
                    continue
                kv = p.split(sep)

                retval[kv[0]] = kv[1]
            return retval

        return pd.DataFrame([_get_params(filename) for filename in filenames])

    df = get_params(input_files)
    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        dg = pd.read_csv(filename)
        dg["filename"] = filename
        dglist += [dg]
    dg = pd.concat(dglist)
    df = pd.merge(df, dg, on="filename")
    return df


def to_numeric(df, to_int, to_float):
    df = df.astype({k: float for k in to_int + to_float}, errors="ignore")
    df = df.astype({k: int for k in to_int}, errors="ignore")
    return df


if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
else:
    input_dir = "../../data/multi_partition_model/evaluations/"
    output_file = "../data/"

#%% Load
data_table = load_files(input_files).fillna("")

# %% Type conversion
to_int = ["n", "K", "dim", "sample", "length", "dim", "cave"]
to_float = ["mu"]
data_table = to_numeric(data_table, to_int, to_float)
data_table = data_table.rename(columns={"K": "q"})

# %% Save
data_table.to_csv(output_file)
