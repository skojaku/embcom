import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace


configfile: "workflow/config.yaml"


include: "./utils.smk"


# =====
# Global
# =====

DATA_DIR = config["data_dir"]
RES_DIR = j(DATA_DIR, "results")

PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

FIG_DIR = j("figs", "{data}")
NET_DIR = j(DATA_DIR, "{data}", "networks")
EMB_DIR = j(DATA_DIR, "{data}", "embedding")
COM_DIR = j(DATA_DIR, "{data}", "communities")
EVA_DIR = j(DATA_DIR, "{data}", "evaluations")
VAL_SPEC_DIR = j(DATA_DIR, "{data}", "spectral_analysis")

# All results
EVAL_CONCAT_FILE = j(EVA_DIR, f"all-result.csv")

# ==========
# Parameters
# ==========

# Embedding
emb_params = {
    "model_name": [
        "node2vec",
        "deepwalk",
        "line",
        "leigenmap",
        "modspec",
        "linearized-node2vec",
        "non-backtracking-node2vec",
        "nonbacktracking",
        "depthfirst-node2vec",
    ],
    "window_length": [10],
    "dim": [64],
}

# Community detection
com_detect_params = {
    "model_name": ["infomap", "flatsbm"],
}

# Clustering
clustering_params = {
    "metric": ["cosine"],
    "clustering": ["voronoi", "kmeans"],
}

# ============
# Data specific
# ============


include: "./multipartition_files.smk"


include: "./lfr_files.smk"


# ======
# RULES
# ======

DATA_LIST = ["lfr"]
# DATA_LIST = ["multi_partition_model", "lfr"]


rule all:
    input:
        expand(EVAL_CONCAT_FILE, data=DATA_LIST),


#        expand(EVAL_FILE, data="multi_partition_model", **net_params, **com_detect_params, **eval_params),
#        expand(EVAL_EMB_FILE, data="multi_partition_model", **net_params, **emb_params, **clustering_params, **eval_params),
#        expand(EMB_FILE, data="multi_partition_model", **net_params, **emb_params),
#        expand(COM_DETECT_FILE, data="multi_partition_model", **net_params, **com_detect_params),
#        expand(COM_DETECT_EMB_FILE, data="multi_partition_model", **net_params, **emb_params, **clustering_params)


rule figs:
    input:
        expand(FIG_PERFORMANCE_VS_MIXING, **fig_params_perf_vs_mixing), #expand(FIG_SPECTRAL_DENSITY_FILE, **bipartition_params)
