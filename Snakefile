import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace


configfile: "workflow/config.yaml"


include: "./utils.smk"
include: "./multipartition_files.smk"

# ======
# RULES
# ======


rule all:
    input:
        #expand(SPECTRAL_DENSITY_FILE, **bipartition_params), #expand(EVAL_FILE, **net_params, **com_detect_params),
        expand(EVAL_FILE, **net_params, **com_detect_params, **eval_params),
        expand(EVAL_EMB_FILE, **net_params, **emb_params, **clustering_params, **eval_params),
        expand(EMB_FILE, **net_params, **emb_params),
        EVAL_CONCAT_FILE,
        expand(COM_DETECT_FILE, **net_params, **com_detect_params),
        expand(COM_DETECT_EMB_FILE, **net_params, **emb_params, **clustering_params)

rule figs:
    input:
        expand(FIG_PERFORMANCE_VS_MIXING, **fig_params_perf_vs_mixing)
        #expand(FIG_SPECTRAL_DENSITY_FILE, **bipartition_params)

#
# network generation
#
rule generate_net_multi_partition_model:
    params:
        parameters=net_paramspace.instance,
    output:
        output_file=NET_FILE,
        output_node_file=NODE_FILE,
    script:
        "workflow/net_generator/generate-net-by-multi-partition-model.py"


#
# Embedding
#
rule embedding_multi_partition_model:
    input:
        net_file=NET_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EMB_FILE,
    params:
        parameters=emb_paramspace.instance,
    script:
        "workflow/embedding/embedding.py"


#
# Clustering
#
rule voronoi_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
    wildcard_constraints:
        clustering="voronoi",
    script:
        "workflow/community-detection/voronoi-clustering.py"


rule kmeans_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
    wildcard_constraints:
        clustering="kmeans",
    script:
        "workflow/community-detection/kmeans-clustering.py"


rule community_detection_multi_partition_model:
    input:
        net_file=NET_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_FILE,
    params:
        parameters=com_detect_paramspace.instance,
    script:
        "workflow/community-detection/detect-community.py"


#
# Evaluation
#
rule evaluate_communities:
    input:
        detected_group_file=COM_DETECT_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_FILE,
    params:
        parameters=eva_paramspace.instance,
    script:
        "workflow/evaluation/eval-com-detect-score.py"

rule evaluate_communities_for_embedding:
    input:
        detected_group_file=COM_DETECT_EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_EMB_FILE,
    params:
        parameters=eva_paramspace.instance,
    script:
        "workflow/evaluation/eval-com-detect-score.py"

#rule concatenate_results:
#    input:
#        input_files = expand(EVAL_FILE, **net_params, **com_detect_params, **eval_params) + expand(EVAL_EMB_FILE, **net_params, **emb_params, **clustering_params, **eval_params)
#    output:
#        output_file=EVAL_CONCAT_FILE
#    script:
#        "workflow/evaluation/concatenate_results.py"

#
# Validating the detectability condition
#
rule calc_spectral_density_linearized_node2vec:
    output:
        output_file=SPECTRAL_DENSITY_FILE,
    params:
        parameters=bipartition_paramspace.instance,
    script:
        "workflow/spectral-density-analysis/calc-spec-density-node2vec.py"


#
# Plot
#
rule plot_performance_vs_mixing: 
    input:
        input_file=EVAL_CONCAT_FILE
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING
    params:
        parameters = fig_perf_vs_mixing_paramspace.instance
    script:
        "workflow/plot/plot-mixing-vs-performance.py"



rule plot_spectral_density:
    input:
        input_file=SPECTRAL_DENSITY_FILE,
    output:
        output_file=FIG_SPECTRAL_DENSITY_FILE
    script:
        "workflow/plot/plot-spectral-density.py"
