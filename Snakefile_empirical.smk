# =========
# FIGURES
# =========
EMP_DIR = j(DATA_DIR, "empirical")
NET_EMP_DIR = j(EMP_DIR, "networks")
net_emp_params = {
    "netdata":["polblog", "cora", "airport"]
}
net_emp_paramspace = to_paramspace(net_emp_params)
NET_EMP_FILE = j(NET_EMP_DIR, f"net_{net_emp_paramspace.wildcard_pattern}.npz")
NODE_EMP_FILE = j(NET_EMP_DIR, f"node_{net_emp_paramspace.wildcard_pattern}.npz")

# =================
# Embedding
# =================
N_SAMPLES_EMP = 10

EMB_EMP_DIR = j(EMP_DIR, "embedding")
emb_emp_paramspace = to_paramspace([net_emp_params, emb_params])
EMB_EMP_FILE = j(EMB_EMP_DIR, f"{emb_emp_paramspace.wildcard_pattern}"+"~sample~{sample}.npz")

# ===================
# Community detection
# ===================
COM_EMP_DIR = j(EMP_DIR, "communities")
com_detect_emp_paramspace = to_paramspace([net_emp_params, com_detect_params])

# Community detection
COM_DETECT_EMP_FILE = j(COM_EMP_DIR, f"{com_detect_emp_paramspace.wildcard_pattern}"+"~sample~{sample}.npz")

# Community detection by clustering to embedding
com_detect_emb_emp_paramspace = to_paramspace([net_emp_params, emb_params, clustering_params])
COM_DETECT_EMB_EMP_FILE = j(
    COM_EMP_DIR, f"clus_{com_detect_emb_emp_paramspace.wildcard_pattern}"+"~sample~{sample}.npz"
)

# =================
# Evaluation
# =================
EVA_EMP_DIR = j(EMP_DIR, "evaluations")
EVAL_EMP_FILE = j(EVA_EMP_DIR, f"score_clus_{com_detect_emp_paramspace.wildcard_pattern}"+"~sample~{sample}.csv")
EVAL_EMB_EMP_FILE = j(EVA_EMP_DIR, f"score_{com_detect_emb_emp_paramspace.wildcard_pattern}"+"~sample~{sample}.csv")

# ======
# RULES
# ======
#
# network generation
#
rule generate_empirical_network:
    output:
        output_file=NET_EMP_FILE,
        output_node_file=NODE_EMP_FILE,
    params:
        parameters=net_emp_paramspace.instance,
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/net_generator/generate-empirical-network.py"


#
# Embedding
#
rule embedding_empirical_network:
    input:
        net_file=NET_EMP_FILE,
        com_file=NODE_EMP_FILE,
    output:
        output_file=EMB_EMP_FILE,
    params:
        parameters=emb_paramspace.instance,
    script:
        "workflow/embedding/embedding.py"


#
# Clustering
#
rule voronoi_clustering_empirical_network:
    input:
        emb_file=EMB_EMP_FILE,
        com_file=NODE_EMP_FILE,
    output:
        output_file=COM_DETECT_EMB_EMP_FILE,
    params:
        parameters=com_detect_emb_emp_paramspace.instance,
    wildcard_constraints:
        clustering="voronoi",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/voronoi-clustering.py"


rule kmeans_clustering_empirical_network:
    input:
        emb_file=EMB_EMP_FILE,
        com_file=NODE_EMP_FILE,
        net_file=NET_EMP_FILE,
    output:
        output_file=COM_DETECT_EMB_EMP_FILE,
    params:
        parameters=com_detect_emb_emp_paramspace.instance,
    wildcard_constraints:
        clustering="kmeans",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/kmeans-clustering.py"


rule eigengapKmeans_clustering_empirical_network:
    input:
        emb_file=EMB_EMP_FILE,
        com_file=NODE_EMP_FILE,
        net_file=NET_EMP_FILE,
    output:
        output_file=COM_DETECT_EMB_EMP_FILE,
    params:
        parameters=com_detect_emb_emp_paramspace.instance,
    wildcard_constraints:
        clustering="eigengap-kmeans",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/eigengap-kmeans.py"

rule knnMod_clustering_empirical_network:
    input:
        emb_file=EMB_EMP_FILE,
        com_file=NODE_EMP_FILE,
    output:
        output_file=COM_DETECT_EMB_EMP_FILE,
    params:
        parameters=com_detect_emb_emp_paramspace.instance,
    wildcard_constraints:
        clustering="knnMod",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/modularity-clustering.py"


rule non_backtracking_kmeans_empirical_network:
    input:
        emb_file=EMB_EMP_FILE,
        com_file=NODE_EMP_FILE,
        net_file=NET_EMP_FILE,
    output:
        output_file=COM_DETECT_EMB_EMP_FILE,
    params:
        parameters=com_detect_emb_emp_paramspace.instance,
    wildcard_constraints:
        clustering="nonbacktrack-kmeans",
    resources:
        mem="12G",
        time="01:00:00"
    script:
        "workflow/community-detection/non-backtracking-kmeans.py"

#
rule community_detection_empirical_network:
    input:
        net_file=NET_EMP_FILE,
        com_file=NODE_EMP_FILE,
    output:
        output_file=COM_DETECT_EMP_FILE,
    params:
        parameters=com_detect_emp_paramspace.instance,
    script:
        "workflow/community-detection/detect-community.py"
#
#
##
## Evaluation
##
rule evaluate_empirical_communities:
    input:
        detected_group_file=COM_DETECT_EMP_FILE,
        com_file=NODE_EMP_FILE,
    output:
        output_file=EVAL_EMP_FILE,
    resources:
        mem="12G",
        time="00:10:00"
    script:
        "workflow/evaluation/eval-com-detect-score.py"
#
#
rule evaluate_empirical_communities_for_embedding:
    input:
        detected_group_file=COM_DETECT_EMB_EMP_FILE,
        com_file=NODE_EMP_FILE,
    output:
        output_file=EVAL_EMB_EMP_FILE,
    resources:
        mem="12G",
        time="00:20:00"
    script:
        "workflow/evaluation/eval-com-detect-score.py"
#
#
#rule concatenate_results_multipartition:
#    input:
#        input_files=expand(
#            EVAL_FILE,
#            data="multi_partition_model",
#            **net_params,
#            **com_detect_params,
#        ) + expand(
#            EVAL_EMB_FILE,
#            data="multi_partition_model",
#            **net_params,
#            **emb_params,
#            **clustering_params,
#        ),
#    output:
#        output_file=EVAL_CONCAT_FILE,
#    params:
#        to_int=["n", "K", "dim", "sample", "length", "dim", "cave"],
#        to_float=["mu"],
#    wildcard_constraints:
#        data="multi_partition_model",
#    resources:
#        mem="4G",
#        time="00:50:00"
#    script:
#        "workflow/evaluation/concatenate_results.py"
#
#
##
## Validating the detectability condition
##
#rule calc_spectral_density_linearized_node2vec:
#    output:
#        output_file=SPECTRAL_DENSITY_FILE,
#    params:
#        parameters=bipartition_paramspace.instance,
#    script:
#        "workflow/spectral-density-analysis/calc-spec-density-node2vec.py"
#
#
##
## Plot
##
#rule plot_performance_vs_mixing:
#    input:
#        #input_file=EVAL_CONCAT_FILE,
#        input_file="data/multi_partition_model/all-result.csv",
#    output:
#        output_file=FIG_PERFORMANCE_VS_MIXING,
#    params:
#        parameters=fig_perf_vs_mixing_paramspace.instance,
#        dimThreshold= False,
#        normalize= False,
#        model_names = ["node2vec", "deepwalk", "line", "modspec", "modspec2", "leigenmap", "nonbacktracking", "bp", "infomap", "flatsbm" ],
#        #title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items()]),
#        with_legend = lambda wildcards: "True" if str(wildcards.cave)=="5" else "False"
#    resources:
#        mem="4G",
#        time="00:50:00"
#    script:
#        "workflow/plot/plot-mixing-vs-performance.py"
#
#rule plot_performance_vs_mixing_mod_vs_spec:
#    input:
#        input_file="data/multi_partition_model/all-result.csv",
#        #input_file=EVAL_CONCAT_FILE,
#    output:
#        output_file=FIG_PERFORMANCE_VS_MIXING_SPEC_VS_SGD,
#    params:
#        parameters=fig_perf_vs_mixing_paramspace.instance,
#        dimThreshold= False,
#        normalize= False,
#        model_names = ["node2vec", "modspec", "modspec2", "leigenmap", "torch-modularity", "torch-laplacian-eigenmap", "bp"],
#        #title = lambda wildcards: " | ".join([f"{k}~{v}" for k, v in wildcards.items()]),
#        with_legend = lambda wildcards: "True" if str(wildcards.cave)=="5" else "False"
#    resources:
#        mem="4G",
#        time="00:50:00"
#    script:
#        "workflow/plot/plot-mixing-vs-performance.py"
#
#
#rule plot_spectral_density:
#    input:
#        input_file=SPECTRAL_DENSITY_FILE,
#    output:
#        output_file=FIG_SPECTRAL_DENSITY_FILE,
#    resources:
#        mem="4G",
#        time="00:50:00"
#    script:
#        "workflow/plot/plot-spectral-density.py"
#
#rule plot_performance_vs_mixing_all:
#    input:
#        expand(FIG_PERFORMANCE_VS_MIXING, **fig_params_perf_vs_mixing),
#    output:
#        output_file=FIG_PERFORMANCE_VS_MIXING_ALL.format(data="multi_partition_model"),
#    run:
#        shell("pdfjam {input} --nup 3x4 --suffix 3up --outfile {output}")
#
#rule plot_loss_landscape_modularity:
#    input:
#        net_file = NET_FILE.format(**params_loss_landscape),
#        node_table_file = NODE_FILE.format(**params_loss_landscape)
#    output:
#        output_file = FIG_LOSS_LANDSCAPE,
#    params:
#        **params_loss_landscape
#    wildcard_constraints:
#        model = "modularity"
#    script:
#        "workflow/plot/plot_modularity_loss_landscape.py"
#
#rule plot_loss_landscape_laplacian:
#    input:
#        net_file = NET_FILE.format(**params_loss_landscape),
#        node_table_file = NODE_FILE.format(**params_loss_landscape)
#    output:
#        output_file = FIG_LOSS_LANDSCAPE,
#    params:
#        **params_loss_landscape
#    wildcard_constraints:
#        model = "laplacian"
#    script:
#        "workflow/plot/plot_laplacian_loss_landscape.py"
#