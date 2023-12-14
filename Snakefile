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
EVAL_CONCAT_ROBUSTNESS_FILE = j(EVA_DIR, f"all-result-robustness.csv")

# ==========
# Parameters
# ==========

# Embedding
emb_params = {
    "model_name": [
        #"node2vec",
        #"deepwalk",
        #"glove",
        #"line",
        #"leigenmap",
        #"modspec",
        "modspec2",
        #"linearized-node2vec",
        #"nonbacktracking",
        #"torch-modularity",
        #"torch-node2vec",
        #"non-backtracking-node2vec",
        #"non-backtracking-deepwalk",
        #"non-backtracking-glove",
        #"depthfirst-node2vec",
    ],
    "window_length": [10],
    "dim": [64],
}

# Community detection
com_detect_params = {
    "model_name": ["bp"],
    #"model_name": ["infomap", "flatsbm", "bp"],
}

# Clustering
clustering_params = {
    "metric": ["cosine"],
    "clustering": ["voronoi", "kmeans"],
}

# ============
# Data specific
# ============

FIG_PERFORMANCE_VS_MIXING_ALL = j(
    FIG_DIR,
    "all_perf_vs_mixing.pdf",
)

include: "./Snakefile_multipartition_files.smk"
include: "./Snakefile_lfr_files.smk"

include: "./Snakefile_robustness_check.smk"


# ======
# RULES
# ======

DATA_LIST = ["multi_partition_model"]
DATA_LIST = ["multi_partition_model", "lfr"]


rule all:
    input:
        expand(EVAL_EMB_FILE, data="multi_partition_model", **net_params, **emb_params, **clustering_params),
        expand(EMB_FILE, data="multi_partition_model", **net_params, **emb_params),
        expand(LFR_EVAL_EMB_FILE, data="lfr", **lfr_net_params, **emb_params, **clustering_params),
        expand(LFR_EMB_FILE, data="lfr", **lfr_net_params, **emb_params),
        #
        # Robustness check
        #
        expand(EVAL_ROBUSTNESS_EMB_FILE , data="multi_partition_model", **robustness_net_params, **robustness_emb_params, **clustering_params),
        expand(EVAL_ROBUSTNESS_LFR_EMB_FILE, data="lfr", **robustness_lfr_net_params, **robustness_emb_params, **clustering_params),
        #expand(COM_DETECT_FILE, data="multi_partition_model", **net_params, **com_detect_params),
        #expand(COM_DETECT_EMB_FILE, data="multi_partition_model", **net_params, **emb_params, **clustering_params)


rule figs:
    input:
        expand(FIG_PERFORMANCE_VS_MIXING, **fig_params_perf_vs_mixing),# expand(FIG_SPECTRAL_DENSITY_FILE, **bipartition_params)
        expand(FIG_PERFORMANCE_VS_MIXING_SPEC_VS_SGD, **fig_params_perf_vs_mixing),# expand(FIG_SPECTRAL_DENSITY_FILE, **bipartition_params)
        #expand(FIG_PERFORMANCE_VS_MIXING_NB, **fig_params_perf_vs_mixing), #expand(FIG_SPECTRAL_DENSITY_FILE, **bipartition_params)
        expand(FIG_LFR_PERFORMANCE_VS_MIXING, **fig_lfr_params_perf_vs_mixing),
        expand(FIG_PERFORMANCE_VS_MIXING_ALL, data = DATA_LIST),
        expand(FIG_LOSS_LANDSCAPE, model = LOSS_LANDSCAPE_MODEL_LIST, )
