"""Module for embedding."""
import logging

import faiss
import gensim
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scipy
from scipy import sparse, stats

from embcom import rsvd, samplers, utils

logger = logging.getLogger(__name__)


try:
    import glove
except ImportError:
    print(
        "Ignore this message if you do not use Glove. Otherwise, install glove python package by 'pip install glove_python_binary' "
    )

#
# Base class
#
class NodeEmbeddings:
    """Super class for node embedding class."""

    def __init__(self):
        self.in_vec = None
        self.out_vec = None

    def fit(self):
        """Estimating the parameters for embedding."""
        pass

    def transform(self, dim, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if self.out_vec is None:
            self.update_embedding(dim)
        elif self.out_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        """Update embedding."""
        pass


class Glove(NodeEmbeddings):
    def __init__(
        self,
        num_walks=10,
        walk_length=40,
        window_length=10,
        restart_prob=0,
        p=1.0,
        q=1.0,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.sampler = samplers.SimpleWalkSampler(
            num_walks,
            walk_length,
            window_length,
            restart_prob,
            p,
            q,
            sample_center_context_pairs=True,
            verbose=False,
        )
        self.learning_rate = 0.05
        self.w2vparams = {"epochs": 25, "no_threads": 4}

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        center, context, freq = self.sampler.get_center_context_pairs()
        center = center.astype(int)
        context = context.astype(int)
        N = self.sampler.num_nodes
        self.cooccur = sparse.coo_matrix(
            (freq, (center, context)), shape=(N, N), dtype="double"
        )
        return self

    def transform(self, dim, return_out_vector=False):
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that
        # for the previous call of transform function
        update_embedding = False
        if self.out_vec is None:
            update_embedding = True
        elif self.out_vec.shape[1] != dim:
            update_embedding = True

        # Update the dimension and train the model
        if update_embedding:
            self.model = glove.Glove(
                no_components=dim, learning_rate=self.learning_rate
            )
            self.model.fit(self.cooccur, **self.w2vparams)
            self.in_vec = self.model.word_vectors
            self.out_vec = self.model.word_vectors

        if return_out_vector:
            return self.out_vec
        else:
            return self.in_vec


class Node2Vec(NodeEmbeddings):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
    restart_prob : float (optional, default 0)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(
        self,
        num_walks=10,
        walk_length=40,
        window_length=10,
        restart_prob=0,
        p=1.0,
        q=1.0,
        verbose=False,
        random_teleport=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.sampler = samplers.SimpleWalkSampler(
            num_walks,
            walk_length,
            window_length,
            restart_prob,
            p,
            q,
            sample_center_context_pairs=False,
            verbose=False,
            random_teleport=random_teleport,
        )

        self.sentences = None
        self.model = None
        self.verbose = verbose

        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": 1,
            "workers": 4,
        }

    def fit(self, net):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        return self

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        self.w2vparams["window"] = self.sampler.window_length

        self.sentences = utils.walk2gensim_sentence(
            self.sampler.walks, self.sampler.window_length
        )

        self.w2vparams["vector_size"] = dim
        self.model = gensim.models.Word2Vec(sentences=self.sentences, **self.w2vparams)

        num_nodes = len(self.model.wv.index_to_key)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if "%d" % i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv["%d" % i]
            self.out_vec[i, :] = self.model.syn1neg[
                self.model.wv.key_to_index["%d" % i]
            ]


class DeepWalk(Node2Vec):
    def __init__(self, **params):
        Node2Vec.__init__(self, **params)
        self.w2vparams = {
            "sg": 0,
            "hs": 1,
            "min_count": 0,
            "workers": 4,
        }


class LaplacianEigenMap(NodeEmbeddings):
    def __init__(self):
        self.in_vec = None
        self.L = None
        self.deg = None

    def fit(self, G):
        A = utils.to_adjacency_matrix(G)

        # Compute the (inverse) normalized laplacian matrix
        deg = np.array(A.sum(axis=1)).reshape(-1)
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        L = Dsqrt @ A @ Dsqrt

        self.L = L
        self.deg = deg
        return self

    def transform(self, dim, return_out_vector=False):
        if self.in_vec is None:
            self.update_embedding(dim)
        elif self.in_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.in_vec

    def update_embedding(self, dim):
        u, s, _ = rsvd.rSVD(self.L, dim + 1)  # add one for the trivial solution
        order = np.argsort(s)[::-1][1:]
        u = u[:, order]

        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(self.deg), 1e-12), format="csr")
        self.in_vec = Dsqrt @ u
        self.out_vec = u


class AdjacencySpectralEmbedding(NodeEmbeddings):
    def __init__(
        self, verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        return self

    def update_embedding(self, dim):
        u, s, v = rsvd.rSVD(self.A, dim=dim)
        self.in_vec = u @ sparse.diags(s)


class ModularitySpectralEmbedding(NodeEmbeddings):
    def __init__(
        self, verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        Q = [
            [self.A],
            [-self.deg.reshape((-1, 1)) / np.sum(self.deg), self.deg.reshape((1, -1))],
        ]
        u, s, v = rsvd.rSVD(Q, dim=dim)
        self.in_vec = u @ sparse.diags(s)
        self.out_vec = None
