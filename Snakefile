import numpy as np
from os.path import join as j


configfile: "workflow/config.yaml"


DATA_DIR = config["data_dir"]
FIG_DIR = "figs"
RES_DIR = j(DATA_DIR, "results")

PAPER_DIR = config["paper_dir"]
PAPER_SRC, SUPP_SRC = [j(PAPER_DIR, f) for f in ("main.tex", "supp.tex")]
PAPER, SUPP = [j(PAPER_DIR, f) for f in ("main.pdf", "supp.pdf")]

DERIVED_DIR = j(DATA_DIR, "derived")
SIM_R_DIR = j(DERIVED_DIR, "sim_R")
SIM_R_RES = j(SIM_R_DIR, "rvals.csv")

rule all:
    input:
        PAPER,
        SUPP,


rule paper:
    input:
        PAPER_SRC,
        SUPP_SRC,
    params:
        paper_dir=PAPER_DIR,
    output:
        PAPER,
        SUPP,
    shell:
        "cd {params.paper_dir}; make"


rule sample_Wij_entry:
    params:
        cin=30,
        cout=5,
        num_sample=100,
    output:
        output_file=SIM_R_RES,
    script:
        "workflow/simulate-R-matrix.py"


# ================================
# Multiple fixed-sized communities
# ================================
MULTI_FIXED_SZ_COM_NET_DIR = j(DATA_DIR, "networks", "multi_fixed_sz_coms")
MULTI_FIXED_SZ_COM_EMB_DIR = j(DATA_DIR, "embeddings", "multi_fixed_sz_coms")
sim_net_params = {
    "n": [1000, 2500, 5000, 7500, 10000],
    "nc": [100],
    "cave": [50],
    "cdiff": [20, 30, 40, 80, 160, 320, 640], # cin - cout
    "sample": np.arange(10),
}
SIM_MULTI_FIXED_SZ_COM_NET = j(
    MULTI_FIXED_SZ_COM_NET_DIR, "net_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}.npz"
)
SIM_MULTI_FIXED_SZ_COM_NET_ALL = expand(SIM_MULTI_FIXED_SZ_COM_NET, **sim_net_params)

# Embedding
MULTI_FIXED_SZ_COM_EMB_FILE_DIR = j(MULTI_FIXED_SZ_COM_EMB_DIR, "embeddings")
emb_params_rw = {  # parameter for methods baesd on random walks
    "model_name": ["node2vec", "glove"],
    "window_length": [3, 5, 10],
    "dim": [1, 64],
}
emb_params = {
    #"model_name": [],
    "model_name": ["leigenmap", "modspec"],
    "window_length": [10],
    "dim": [1, 64],
}
MULTI_FIXED_SZ_COM_EMB_FILE = j(
    MULTI_FIXED_SZ_COM_EMB_FILE_DIR,
    "embnet_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.npz",
)
MULTI_FIXED_SZ_COM_EMB_FILE_ALL = expand(
    MULTI_FIXED_SZ_COM_EMB_FILE, **sim_net_params, **emb_params
) + expand(MULTI_FIXED_SZ_COM_EMB_FILE, **sim_net_params, **emb_params_rw)

# Community
MULTI_FIXED_SZ_COM_DIR = j(MULTI_FIXED_SZ_COM_EMB_DIR, "communities")
MULTI_FIXED_SZ_COM_FILE = j(
    MULTI_FIXED_SZ_COM_DIR,
    "community_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}.npz",
)
com_detect_params = {
    "model_name": ["infomap"],
}
MULTI_FIXED_SZ_COM_FILE_ALL = expand(
    MULTI_FIXED_SZ_COM_FILE, **sim_net_params, **com_detect_params
)

# Derived
MULTI_FIXED_SZ_COM_AUC_FILE =        j(RES_DIR, "multi_fixed_size_coms", "auc", "auc_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv")
MULTI_FIXED_SZ_COM_SIM_FILE =        j(RES_DIR, "multi_fixed_size_coms", "similarity", "similarity_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv")
MULTI_FIXED_SZ_COM_KMEANS_FILE = j(RES_DIR, "multi_fixed_size_coms", "kmeans", "kmeans_n={n}_nc={nc}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv")

MULTI_FIXED_SZ_COM_AUC_FILE_ALL = expand(MULTI_FIXED_SZ_COM_AUC_FILE, **sim_net_params, **emb_params) + expand(MULTI_FIXED_SZ_COM_AUC_FILE, **sim_net_params, **emb_params_rw)
MULTI_FIXED_SZ_COM_SIM_FILE_ALL = expand(MULTI_FIXED_SZ_COM_SIM_FILE, **sim_net_params, **emb_params) + expand(MULTI_FIXED_SZ_COM_SIM_FILE, **sim_net_params, **emb_params_rw)
MULTI_FIXED_SZ_COM_KMEANS_FILE_ALL = expand(MULTI_FIXED_SZ_COM_KMEANS_FILE, **sim_net_params, **emb_params) + expand(MULTI_FIXED_SZ_COM_KMEANS_FILE, **sim_net_params, **emb_params_rw)

MULTI_FIXED_SZ_COM_AUC_RES_FILE  = j(RES_DIR, "multi_fixed_size_coms", "results", "auc.csv")
MULTI_FIXED_SZ_COM_KMEANS_RES_FILE = j(RES_DIR, "multi_fixed_size_coms", "results", "kmeans.csv")

rule generate_fixed_size_multi_com_net:
    params:
        cave=lambda wildcards: int(wildcards.cave),
        cdiff=lambda wildcards: int(wildcards.cdiff),
        n=lambda wildcards: int(wildcards.n),
        nc=lambda wildcards: int(wildcards.nc),
    output:
        output_file=SIM_MULTI_FIXED_SZ_COM_NET,
    script:
        "workflow/generate-multi-fixed-size-com-net.py"

rule multi_fixed_size_com_embedding:
    input:
        netfile=SIM_MULTI_FIXED_SZ_COM_NET,
    output:
        embfile=MULTI_FIXED_SZ_COM_EMB_FILE,
    params:
        model_name=lambda wildcards: wildcards.model_name,
        dim=lambda wildcards: wildcards.dim,
        window_length=lambda wildcards: wildcards.window_length,
        directed="undirected",
        num_walks=5,
    script:
        "workflow/embedding.py"

rule eval_auc_fixed_size_com_embedding:
    input:
        emb_files=MULTI_FIXED_SZ_COM_EMB_FILE,
    params:
        K = lambda wildcards : int(wildcards.n) / int(wildcards.nc)
    output:
        output_file=MULTI_FIXED_SZ_COM_AUC_FILE,
        output_sim_file=MULTI_FIXED_SZ_COM_SIM_FILE,
    script:
        "workflow/eval-community.py"


rule eval_fixed_size_com_embedding_kmeans:
    input:
        emb_files=MULTI_FIXED_SZ_COM_EMB_FILE,
    params:
        K = lambda wildcards : int(wildcards.n) / int(wildcards.nc)
    output:
        output_sim_file=MULTI_FIXED_SZ_COM_KMEANS_FILE,
    script:
        "workflow/eval-community-kmeans.py"

rule concat_auc_result_fixed_size_file:
    input:
        input_files = MULTI_FIXED_SZ_COM_AUC_FILE_ALL
    output:
        output_file = MULTI_FIXED_SZ_COM_AUC_RES_FILE
    script:
        "workflow/concat-files.py"

rule concat_kmeans_result_fixed_size_file:
    input:
        input_files = MULTI_FIXED_SZ_COM_KMEANS_FILE_ALL
    output:
        output_file = MULTI_FIXED_SZ_COM_KMEANS_RES_FILE
    script:
        "workflow/concat-files.py"

rule detect_fixed_size_community_by_infomap:
    input:
        netfile=SIM_MULTI_FIXED_SZ_COM_NET
    output:
        output_file = MULTI_FIXED_SZ_COM_FILE
    script:
        "workflow/detect-community-by-infomap.py"


# ==================================
# Multiple variable-size communities
# ==================================
MULTI_COM_NET_DIR = j(DATA_DIR, "networks", "multi_coms")
MULTI_COM_EMB_DIR = j(DATA_DIR, "embeddings", "multi_coms")
sim_net_params = {
    "n": [1000, 2500, 5000, 7500, 10000, 100000],
    "cave": [50],
    "cdiff": [20, 30, 40, 80, 160, 320, 640], # cin - cout
    "K": [2, 50],
    "sample": np.arange(10),
}
SIM_MULTI_COM_NET = j(
    MULTI_COM_NET_DIR, "net_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}.npz"
)
SIM_MULTI_COM_NET_ALL = expand(SIM_MULTI_COM_NET, **sim_net_params)

MULTI_COM_EMB_FILE_DIR = j(MULTI_COM_EMB_DIR, "embeddings")
emb_params_rw = {  # parameter for methods baesd on random walks
    "model_name": ["node2vec", "glove"],
    "window_length": [10],
    "dim": [1, 64],
}
emb_params = {
    "model_name": ["leigenmap", "modspec"],
    "window_length": [10],
    "dim": [1, 64],
}
MULTI_COM_EMB_FILE = j(
    MULTI_COM_EMB_FILE_DIR,
    "embnet_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.npz",
)
MULTI_COM_EMB_FILE_ALL = expand(
    MULTI_COM_EMB_FILE, **sim_net_params, **emb_params
) + expand(MULTI_COM_EMB_FILE, **sim_net_params, **emb_params_rw)

# Community detection
MULTI_COM_DIR = j(MULTI_COM_EMB_DIR, "communities")
MULTI_COM_FILE = j(
    MULTI_COM_DIR,
    "community_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}.npz",
)
com_detect_params = {
    "model_name": ["infomap"],
}
MULTI_COM_FILE_ALL = expand(
    MULTI_COM_FILE, **sim_net_params, **com_detect_params
)


# Derived
MULTI_COM_AUC_FILE =       j(RES_DIR, "multi_coms", "auc", "auc_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv")
MULTI_COM_SIM_FILE =       j(RES_DIR, "multi_coms", "similarity", "similarity_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv")
MULTI_COM_KMEANS_FILE =j(RES_DIR, "multi_coms", "kmeans", "kmeans_n={n}_K={K}_cave={cave}_cdiff={cdiff}_sample={sample}_model={model_name}_wl={window_length}_dim={dim}.csv")
MULTI_COM_AUC_FILE_ALL = expand(MULTI_COM_AUC_FILE, **sim_net_params, **emb_params) + expand(MULTI_COM_AUC_FILE, **sim_net_params, **emb_params_rw)
MULTI_COM_SIM_FILE_ALL = expand(MULTI_COM_SIM_FILE, **sim_net_params, **emb_params) + expand(MULTI_COM_SIM_FILE, **sim_net_params, **emb_params_rw)
MULTI_COM_KMEANS_FILE_ALL = expand(MULTI_COM_KMEANS_FILE, **sim_net_params, **emb_params) + expand(MULTI_COM_KMEANS_FILE, **sim_net_params, **emb_params_rw)


MULTI_COM_AUC_RES_FILE  = j(RES_DIR, "multi_coms", "results", "auc.csv")
MULTI_COM_KMEANS_RES_FILE = j(RES_DIR, "multi_coms", "results", "kmeans.csv")

rule generate_multi_com_net:
    params:
        cave=lambda wildcards: int(wildcards.cave),
        cdiff=lambda wildcards: int(wildcards.cdiff),
        n=lambda wildcards: int(wildcards.n),
        K=lambda wildcards: int(wildcards.K),
    output:
        output_file=SIM_MULTI_COM_NET,
    script:
        "workflow/generate-multi-com-net.py"

rule multi_com_embedding:
    input:
        netfile=SIM_MULTI_COM_NET,
    output:
        embfile=MULTI_COM_EMB_FILE,
    params:
        model_name=lambda wildcards: wildcards.model_name,
        dim=lambda wildcards: wildcards.dim,
        window_length=lambda wildcards: wildcards.window_length,
        directed="undirected",
        num_walks=5,
    script:
        "workflow/embedding.py"

rule eval_auc_multi_com_embedding:
    input:
        emb_files=MULTI_COM_EMB_FILE,
    params:
        K = lambda wildcards :wildcards.K
    output:
        output_file=MULTI_COM_AUC_FILE,
        output_sim_file=MULTI_COM_SIM_FILE,
    script:
        "workflow/eval-community.py"


rule eval_multi_com_embedding_kmeans:
    input:
        emb_files=MULTI_COM_EMB_FILE,
    params:
        K = lambda wildcards :wildcards.K
    output:
        output_sim_file=MULTI_COM_KMEANS_FILE,
    script:
        "workflow/eval-community-kmeans.py"

rule concat_auc_result_multi_com_file:
    input:
        input_files = MULTI_COM_AUC_FILE_ALL
    output:
        output_file = MULTI_COM_AUC_RES_FILE
    script:
        "workflow/concat-files.py"

rule concat_kmeans_result_multi_com_file:
    input:
        input_files = MULTI_COM_KMEANS_FILE_ALL
    output:
        output_file = MULTI_COM_KMEANS_RES_FILE
    script:
        "workflow/concat-files.py"

rule detect_community_by_infomap:
    input:
        netfile=SIM_MULTI_COM_NET
    output:
        output_file = MULTI_COM_FILE
    script:
        "workflow/detect-community-by-infomap.py"


#
# Misc
#
rule _all:
    input:
        #MULTI_COM_AUC_RES_FILE,
        #MULTI_COM_KMEANS_RES_FILE,
        MULTI_FIXED_SZ_COM_AUC_RES_FILE,
        MULTI_FIXED_SZ_COM_KMEANS_RES_FILE,
        MULTI_COM_AUC_RES_FILE,
        MULTI_COM_KMEANS_RES_FILE,
        #TWO_COM_AUC_RES_FILE, RES_TWO_COM_KMEANS_RES_FILE,
        #MULTI_FIXED_SZ_COM_AUC_RES_FILE, RES_MULTI_FIXED_SZ_COM_KMEANS_RES_FILE
        #RES_TWO_COM_KMEANS_FILE_ALL,TWO_COM_SIM_FILE_ALL,TWO_COM_AUC_FILE_ALL,
        #RES_MULTI_FIXED_SZ_COM_KMEANS_FILE_ALL,MULTI_FIXED_SZ_COM_SIM_FILE_ALL,MULTI_FIXED_SZ_COM_AUC_FILE_ALL,
        #RES_MULTI_COM_KMEANS_FILE_ALL,MULTI_COM_SIM_FILE_ALL,MULTI_COM_AUC_FILE_ALL,
        #MULTI_FIXED_SZ_COM_AUC_FILE,
        #"MULTI_FIXED_SZ_COM_SIM_FILE,
        #RES_MULTI_FIXED_SZ_COM_KMEANS_FILE,
        #"MULTI_COM_SIM_FILE,
        #RES_MULTI_COM_KMEANS_FILE
        #TWO_COM_EMB_FILE_ALL, #SIM_TWO_COM_NET_ALL
         #TWO_COM_SIM_FILE,RES_TWO_COM_KMEANS_FILE
         #TWO_COM_EMB_FILE_ALL

rule __all:
    input:
        MULTI_FIXED_SZ_COM_FILE_ALL, MULTI_COM_FILE_ALL
        #TWO_COM_EMB_FILE_ALL, #SIM_TWO_COM_NET_ALL
         #TWO_COM_SIM_FILE,RES_TWO_COM_KMEANS_FILE
         #TWO_COM_EMB_FILE_ALL
# rule some_data_processing:
# input:
# "data/some_data.csv"
# output:
# "data/derived/some_derived_data.csv"
# script:
# "workflow/scripts/process_some_data.py"
