#%%
import logging
import sys

import fastnode2vec
import numpy as np
import pandas as pd
from scipy import sparse

import embcom

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

#
# Input
#
if "snakemake" in sys.modules:
    #    input_file = snakemake.input['input_file']
    #    output_file = snakemake.output['output_file']
    netfile = snakemake.input["net_file"]
    com_file = snakemake.input["com_file"]
    embfile = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    dim = int(params["dim"])
    window_length = int(params["window_length"])
    model_name = params["model_name"]
    num_walks = 10
else:
    netfile = "../../data/multi_partition_model/networks/net_n~100000_K~1000_cave~20_mu~0.1_sample~0.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~100000_K~1000_cave~20_mu~0.1_sample~0.npz"
    embfile = "tmp.npz"
    dim = 64
    window_length = 10
    model_name = "line"
    num_walks = 10


net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0 + 1

true_membership = pd.read_csv(com_file)["membership"].values.astype(int)

if dim == 0:
    dim = len(set(true_membership)) - 1
    dim = np.minimum(net.shape[0] - 1, dim)

#
# Embedding models
#
if model_name == "levy-word2vec":
    model = embcom.embeddings.LevyWord2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "node2vec":
    model = fastnode2vec.Node2Vec(window_length=window_length, num_walks=num_walks)
elif model_name == "depthfirst-node2vec":
    model = fastnode2vec.Node2Vec(
        window_length=window_length, num_walks=num_walks, p=10, q=0.1
    )
elif model_name == "node2vec-qhalf":
    model = fastnode2vec.Node2Vec(
        window_length=window_length, num_walks=num_walks, q=0.5
    )
elif model_name == "node2vec-qdouble":
    model = fastnode2vec.Node2Vec(window_length=window_length, num_walks=num_walks, q=2)
elif model_name == "deepwalk":
    model = fastnode2vec.DeepWalk(window_length=window_length, num_walks=num_walks)
elif model_name == "line":
    model = fastnode2vec.LINE(num_walks=num_walks, workers = 4)
elif model_name == "glove":
    model = embcom.embeddings.Glove(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "leigenmap":
    model = embcom.embeddings.LaplacianEigenMap()
elif model_name == "adjspec":
    model = embcom.embeddings.AdjacencySpectralEmbedding()
elif model_name == "modspec":
    model = embcom.embeddings.ModularitySpectralEmbedding()
elif model_name == "nonbacktracking":
    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
elif model_name == "node2vec-matrixfact":
    model = embcom.embeddings.Node2VecMatrixFactorization(
        window_length=window_length, blocking_membership=None
    )
elif model_name == "highorder-modspec":
    model = embcom.embeddings.HighOrderModularitySpectralEmbedding(
        window_length=window_length
    )
elif model_name == "linearized-node2vec":
    model = embcom.embeddings.LinearizedNode2Vec(window_length=window_length)
elif model_name == "non-backtracking-node2vec":
    model = embcom.embeddings.NonBacktrackingNode2Vec(
        window_length=window_length, num_walks=num_walks
    )
elif model_name == "non-backtracking-deepwalk":
    model = embcom.embeddings.NonBacktrackingDeepWalk(
        window_length=window_length, num_walks=num_walks
    )
elif model_name == "non-backtracking-glove":
    model = embcom.embeddings.NonBacktrackingGlove(
        window_length=window_length, num_walks=num_walks
    )

#
# Embedding
#
model.fit(sparse.csr_matrix(net))
emb = model.transform(dim=dim)

#
# Save
#
np.savez_compressed(
    embfile, emb=emb, window_length=window_length, dim=dim, model_name=model_name,
)

# %%
