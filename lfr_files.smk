# ================================
# Networks and communities
# ================================

lfr_net_params = {
    "n": [100000],  # Network size
    "k": [50, 100],  # Average degree
    "tau": [3],  # degree exponent
    "tau2": [1],  # community size exponent
    "minc": [50],  # min community size
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": np.arange(3),  # Number of samples
}

# Convert to a paramspace
lfr_net_paramspace = to_paramspace(lfr_net_params)
LFR_NET_FILE = j(NET_DIR, f"net_{lfr_net_paramspace.wildcard_pattern}.npz")
LFR_NODE_FILE = j(NET_DIR, f"node_{lfr_net_paramspace.wildcard_pattern}.npz")

# =================
# Embedding
# =================
lfr_emb_paramspace = to_paramspace([lfr_net_params, emb_params])
LFR_EMB_FILE = j(EMB_DIR, f"{lfr_emb_paramspace.wildcard_pattern}.npz")

# ===================
# Community detection
# ===================
lfr_com_detect_paramspace = to_paramspace([lfr_net_params, com_detect_params])

# Community detection
LFR_COM_DETECT_FILE = j(COM_DIR, f"{lfr_com_detect_paramspace.wildcard_pattern}.npz")

# Community detection by clustering to embedding
lfr_com_detect_emb_paramspace = to_paramspace([lfr_net_params, emb_params, clustering_params])
LFR_COM_DETECT_EMB_FILE = j(
    COM_DIR, f"clus_{lfr_com_detect_emb_paramspace.wildcard_pattern}.npz"
)


# ==========
# Evaluation
# ==========
lfr_eva_emb_paramspace = to_paramspace([eval_params, lfr_net_params, emb_params, clustering_params])
LFR_EVAL_EMB_FILE = j(EVA_DIR, f"score_clus_{lfr_eva_emb_paramspace.wildcard_pattern}.npz")

lfr_eva_paramspace = to_paramspace([eval_params, lfr_net_params, com_detect_params])
LFR_EVAL_FILE = j(EVA_DIR, f"score_{lfr_eva_paramspace.wildcard_pattern}.npz")

# ======
# RULES
# ======
#
# network generation
#
rule generate_lfr_net:
    params:
        parameters=lfr_net_paramspace.instance,
    output:
        output_file=LFR_NET_FILE,
        output_node_file=LFR_NODE_FILE,
    wildcard_constraints:
        data="lfr"
    resources:
        mem="12G",
        time="04:00:00"
    script:
        "workflow/net_generator/generate-lfr-net.py"

#
# Embedding
#
use rule embedding_multi_partition_model as embedding_lfr with:
    input:
        net_file=LFR_NET_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EMB_FILE,
    params:
        parameters=lfr_emb_paramspace.instance,


#
# Clustering
#
use rule voronoi_clustering_multi_partition_model as voronoi_clustering_lfr with:
    input:
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=lfr_com_detect_emb_paramspace.instance,


use rule kmeans_clustering_multi_partition_model as kmeans_clustering_lfr with:
    input:
        emb_file=LFR_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_EMB_FILE,
    params:
        parameters=lfr_com_detect_emb_paramspace.instance,


use rule community_detection_multi_partition_model as community_detection_lfr with:
    input:
        net_file=LFR_NET_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_COM_DETECT_FILE,
    params:
        parameters=lfr_com_detect_paramspace.instance,


#
# Evaluation
#
use rule evaluate_communities as evaluate_communities_lfr with:
    input:
        detected_group_file=LFR_COM_DETECT_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EVAL_FILE,
    params:
        parameters=lfr_eva_paramspace.instance,

use rule evaluate_communities_for_embedding as evaluate_communities_for_embedding_lfr with:
    input:
        detected_group_file=LFR_COM_DETECT_EMB_FILE,
        com_file=LFR_NODE_FILE,
    output:
        output_file=LFR_EVAL_EMB_FILE,
    params:
        parameters=lfr_eva_paramspace.instance,

rule concatenate_results_lfr:
    input:
        input_files = expand(LFR_EVAL_FILE, data="lfr", **lfr_net_params, **com_detect_params, **eval_params) + expand(LFR_EVAL_EMB_FILE, data="lfr", **lfr_net_params, **emb_params, **clustering_params, **eval_params)
    output:
        output_file=EVAL_CONCAT_FILE
    wildcard_constraints:
        data="lfr"
    script:
        "workflow/evaluation/concatenate_results.py"
