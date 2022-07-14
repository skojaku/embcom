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
from sklearn.decomposition import TruncatedSVD

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
        walk_length=80,
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


class LevyWord2Vec(NodeEmbeddings):
    """A python class for the Levy's matrix MF.

    Equivalent to pResidual2Vec with the configuration sampler
    """

    def __init__(
        self,
        num_walks=10,
        walk_length=80,
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
        logger.debug("sampling - start")
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        logger.debug("sampling - finished")
        return self

    def update_embedding(self, dim):
        # Update the dimension and train the model

        # Sample the sequence of nodes using a random walk
        logger.debug("retrieve center context pairs")
        center, context, freq = self.sampler.get_center_context_pairs()
        center = center.astype(int)
        context = context.astype(int)
        logger.debug("{} pairs generated".format(len(center)))

        logger.debug("count the frequency of words")
        Pi = np.bincount(center, weights=freq, minlength=self.sampler.A.shape[0])
        Pi = np.minimum(Pi, 1)
        logger.debug("Calculate the Q matrix")
        Qij = (
            np.log(np.sum(freq))
            + np.log(freq)
            - np.log(Pi[center])
            - np.log(Pi[context])
        )
        s = Qij > 0
        Q = sparse.csr_matrix(
            (freq[s], (center[s], context[s])), shape=self.sampler.A.shape
        )
        logger.debug("SVD")
        svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
        in_vec = svd.fit_transform(Q)
        val = svd.singular_values_
        out_vec = in_vec.copy()
        # in_vec, val, out_vec = rsvd.rSVD(Q, dim)
        order = np.argsort(val)[::-1]
        val = val[order]
        alpha = 0.5
        self.in_vec = in_vec[:, order] @ np.diag(np.power(val, alpha))
        self.out_vec = out_vec[order, :].T @ np.diag(np.power(val, 1 - alpha))


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
        u, s, v = rsvd.rSVD(self.L, dim + 1)  # add one for the trivial solution
        sign = np.sign(np.diag(v @ u))
        s = s * sign
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
        svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
        u = svd.fit_transform(self.A)
        s = svd.singular_values_
        # u, s, v = rsvd.rSVD(self.A, dim=dim)
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
        sign = np.sign(np.diag(v @ u))
        s = s * sign
        self.in_vec = u @ sparse.diags(s)
        self.out_vec = None


class HighOrderModularitySpectralEmbedding(NodeEmbeddings):
    def __init__(
        self, verbose=False, window_length=10,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        stationary_prob = self.deg / np.sum(self.deg)

        P = utils.to_trans_mat(self.A)
        Q = []
        for t in range(self.window_length):
            Q.append(
                [sparse.diags(stationary_prob / self.window_length) @ P]
                + [P for _ in range(t)]
            )
        Q.append([-stationary_prob.reshape((-1, 1)), stationary_prob.reshape((1, -1))])
        u, s, v = rsvd.rSVD(Q, dim=dim)
        self.in_vec = u @ sparse.diags(s)
        self.out_vec = None


class LinearizedNode2Vec(NodeEmbeddings):
    def __init__(
        self, verbose=False, window_length=10,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):

        # Calculate the normalized transition matrix
        Dinvsqrt = sparse.diags(1 / np.sqrt(np.maximum(1, self.deg)))
        Psym = Dinvsqrt @ self.A @ Dinvsqrt

        svd = TruncatedSVD(n_components=dim + 1, n_iter=7, random_state=42)
        u = svd.fit_transform(Psym)
        s = svd.singular_values_
        # u, s, v = rsvd.rSVD(Psym, dim=dim + 1)
        # sign = np.sign(np.diag(v @ u))
        # s = s * sign
        # mask = s < np.max(s)
        # u = u[:, mask]
        # s = s[mask]

        if self.window_length > 1:
            s = (s * (1 - s ** self.window_length)) / (self.window_length * (1 - s))

        self.in_vec = u @ sparse.diags(s)
        self.out_vec = None


class NonBacktrackingSpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, auto_dim=False):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.auto_dim = auto_dim
        self.C = 10

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        self.A = A
        return self

    def update_embedding(self, dim):

        N = self.A.shape[0]
        Z = sparse.csr_matrix((N, N))
        I = sparse.identity(N, format="csr")
        D = sparse.diags(self.deg)
        B = sparse.bmat([[Z, D - I], [-I, self.A]], format="csr")

        if self.auto_dim is False:
            s, v = sparse.linalg.eigs(B, k=dim + 1, tol=1e-4)
            order = np.argsort(s)
            s, v = s[order], v[:, order]
            s, v = s[1:], v[:, 1:]
            v = v[N:, :]
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)
        else:
            dim = int(self.C * np.sqrt(N))
            dim = np.minimum(dim, N - 1)

            s, v = sparse.linalg.eigs(B, k=dim + 1, tol=1e-4)

            c = int(self.A.sum() / N)
            s, v = s[np.abs(s) > c], v[:, np.abs(s) > c]

            order = np.argsort(s)
            s, v = s[order], v[:, order]
            s, v = s[1:], v[:, 1:]
            v = v[N:, :]
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

        self.in_vec = v


class Node2VecMatrixFactorization(NodeEmbeddings):
    def __init__(self, verbose=False, window_length=10, num_blocks=500):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.num_blocks = num_blocks

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)

        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):

        P = utils.to_trans_mat(self.A)
        Ppow = utils.matrix_sum_power(P, self.window_length) / self.window_length
        stationary_prob = self.deg / np.sum(self.deg)
        R = np.log(Ppow @ np.diag(1 / stationary_prob))

        # u, s, v = rsvd.rSVD(R, dim=dim)
        svd = TruncatedSVD(n_components=dim + 1, n_iter=7, random_state=42)
        u = svd.fit_transform(R)
        s = svd.singular_values_
        self.in_vec = u @ sparse.diags(np.sqrt(s))
        self.out_vec = None


class NonBacktrackingNode2Vec(Node2Vec):
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

    def __init__(self, num_walks=10, walk_length=80, window_length=10, **params):
        Node2Vec.__init__(
            self,
            num_walks=num_walks,
            walk_length=walk_length,
            window_length=window_length,
            **params
        )
        self.sampler = samplers.NonBacktrackingWalkSampler(
            num_walks=num_walks, walk_length=walk_length, window_length=window_length
        )


class NonBacktrackingDeepWalk(DeepWalk):
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

    def __init__(self, num_walks=10, walk_length=80, window_length=10, **params):
        DeepWalk.__init__(
            self,
            num_walks=num_walks,
            walk_length=walk_length,
            window_length=window_length,
            **params
        )
        self.sampler = samplers.NonBacktrackingWalkSampler(
            num_walks=num_walks, walk_length=walk_length, window_length=window_length
        )


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


class NonBacktrackingGlove(Glove):
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

    def __init__(self, num_walks=10, walk_length=80, window_length=10, **params):
        Glove.__init__(
            self,
            num_walks=num_walks,
            walk_length=walk_length,
            window_length=window_length,
            **params
        )
        self.sampler = samplers.NonBacktrackingWalkSampler(
            num_walks=10, walk_length=80, window_length=10
        )
