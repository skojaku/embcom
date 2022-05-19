"""This module contains sampler class which generates a sequence of nodes from
a network using a random walk.

All samplers should be the subclass of NodeSampler class.
"""
import logging
from abc import ABCMeta, abstractmethod
from collections import Counter

import faiss
import numba
import numpy as np
from numba import prange
from numba import njit
from scipy import sparse
from sklearn.cluster import KMeans

from embcom import utils

logger = logging.getLogger(__name__)


class NodeSampler(metaclass=ABCMeta):
    """Super class for node sampler class.

    Implement
        - get_trans_matrix
        - sampling

    Optional
        - get_decomposed_trans_matrix
    """

    @abstractmethod
    def get_trans_matrix(self, scale="normal"):
        """Construct the transition matrix for the node sequence.

        Return
        ------
        trans_prob : sparse.csr_matrix
            Transition matrix. trans_prob[i,j] is the probability
            that a random walker in node i moves to node j.
        """

    def get_trans_prob(self, src, trg):
        """Construct the transition matrix for the node sequence.

        Parameters
        ----------
        src : numpy.ndarray
            IDs of source nodes
        trg : numpy.ndarray
            IDs of target nodes


        Return
        ------
        trans_prob : numpy.ndarray
            Transition probability
        """
        P = self.get_trans_matrix()
        return np.array(P[(src, trg)]).reshape(-1)

    @abstractmethod
    def sampling(self):
        """Generate a sequence of walks over the network.

        Return
        ------
        walks : numpy.ndarray (number_of_walks, number_of_steps)
            Each row indicates a trajectory of a random walker
            walk[i,j] indicates the jth step for walker i.
        """

    @abstractmethod
    def get_center_context_pairs(self, num_walks=5):
        """get center and context pairs."""

    @abstractmethod
    def sample_context(self, pos_pairs, sz):
        """sample context from center."""


#
# SimpleWalk Sampler
#
class SimpleWalkSampler(NodeSampler):
    def __init__(
        self,
        num_walks=10,
        walk_length=80,
        window_length=10,
        restart_prob=0,
        p=1.0,
        q=1.0,
        verbose=False,
        sample_center_context_pairs=True,
        random_teleport=False,
        **params
    ):
        """Simple walk with restart.

        Parameters
        ----------
        num_walks : int (Optional; Default 1)
            Number of walkers to simulate for each randomized network.
            A larger value removes the bias better but takes more time.
        walk_length : int
            Number of steps for a single walker
        p : float
            Parameter for the node2vec
        q : float
            Parameter for the node2vec
        window_length : int
            Size of the window
        restart_prob : float
            Probability of teleport back to the starting node
        verbose : bool
            Set true to display the progress (NOT IMPLEMENTED)
        """

        self.restart_prob = restart_prob
        self.trans_prob = None
        self.verbose = verbose
        self.num_nodes = -1

        # parameters for random walker
        self.num_walks = int(num_walks)
        self.walk_length = walk_length
        self.sample_center_context_pairs = sample_center_context_pairs
        self.p = p
        self.q = q
        self.walks = None
        self.W = None
        self.cum_trans_prob = None

        # parameters for learning embeddings
        self.window_length = window_length
        self.random_teleport = random_teleport

    def _init_center_context_pairs(self, center, context, freq):
        self.W = sparse.csr_matrix(
            (freq, (center, context)), shape=(self.num_nodes, self.num_nodes),
        )
        self.cum_trans_prob = calc_cum_trans_prob(self.W.copy())
        self.center_prob = np.array(self.W.sum(axis=1)).reshape(-1)
        if ~np.isclose(np.sum(self.center_prob), 0):
            self.center_prob /= np.sum(self.center_prob)

    def sampling(self, net):
        self.num_nodes = net.shape[0]
        self.A = net

        if np.isclose(self.p, 1) and np.isclose(self.q, 1):
            if self.sample_center_context_pairs:
                center, context, freq = sample_center_context_pair(
                    self.A,
                    self.num_walks,
                    self.walk_length,
                    self.restart_prob,
                    self.window_length,
                    random_teleport=self.random_teleport,
                )
                self._init_center_context_pairs(center, context, freq)
            else:
                self.walks = simulate_simple_walk(
                    self.A,
                    self.num_walks,
                    self.walk_length,
                    self.restart_prob,
                    random_teleport=self.random_teleport,
                )
                self.walks = self.walks.astype(int)
        else:  # biased walk
            # calc supra-adjacency matrix
            Asupra, node_pairs = utils.construct_line_net_adj(net, p=self.p, q=self.q)
            self.num_nodes_supra = Asupra.shape[0]

            # Find the starting node ids
            start_node_ids = np.where(node_pairs[:, 0] == node_pairs[:, 1])[0]
            self.walks = simulate_simple_walk(
                Asupra,
                self.num_walks,
                self.walk_length,
                self.restart_prob,
                start_node_ids=start_node_ids,
            )
            self.walks = self.walks.astype(int)

            # Convert supra-node id to the id of the nodes in the original net
            self.walks = node_pairs[self.walks.reshape(-1), 1].reshape(self.walks.shape)

    def get_trans_matrix(self, scale="normal"):
        """Compute the transition probability from one node to another.

        Return
        ------
        trans_prob : numpy.ndarray
            Transition probability matrix of size.
            (number of nodes, number of nodes).
            trans_prob[i,j] indicates the transition probability
            from node i to node j.
        """

        # Generate a sequence of random walks
        if self.W is None:
            self.get_center_context_pairs()
        if self.joint_prob is False:
            trans_prob = utils.to_trans_mat(self.W)
        if scale == "log":
            trans_prob.data = utils.safe_log(trans_prob.data)
        return trans_prob

    def get_center_context_pairs(self):
        if self.W is None:
            (center, context, freq,) = generate_center_context_pair(
                self.walks, self.window_length
            )
            self._init_center_context_pairs(center, context, freq)
            return center, context, freq
        else:
            center, context, freq = sparse.find(self.W)
            return center, context, freq

    def sample_context(self, center, sz):
        context = sample_columns_from_cum_prob(
            np.repeat(center, sz),
            self.cum_trans_prob.indptr,
            self.cum_trans_prob.indices,
            self.cum_trans_prob.data,
        )
        context = context.reshape((len(center), sz))
        return context


#
# Non-backtracking walks
#
class NonBacktrackingWalkSampler(NodeSampler):
    def __init__(
        self, num_walks=10, walk_length=80, window_length=10, verbose=False, **params
    ):
        self.verbose = verbose
        self.num_nodes = -1
        self.num_walks = int(num_walks)
        self.walk_length = walk_length
        self.window_length = window_length
        self.walks = None

    def sampling(self, net):
        self.num_nodes = net.shape[0]
        self.A = net
        self.walks = simulate_non_backtracking_walk(
            self.A, self.num_walks, self.walk_length,
        )
        self.walks = self.walks.astype(int)

    def get_trans_matrix(self, scale="normal"):
        """Construct the transition matrix for the node sequence.

        Return
        ------
        trans_prob : sparse.csr_matrix
            Transition matrix. trans_prob[i,j] is the probability
            that a random walker in node i moves to node j.
        """
        raise NotImplementedError("")

    def get_center_context_pairs(self, num_walks=5):
        raise NotImplementedError("")

    def sample_context(self, pos_pairs, sz):
        raise NotImplementedError("")


def simulate_non_backtracking_walk(
    A, num_walk, walk_length, start_node_ids=None, is_cum_trans_prob_mat=False, **params
):
    """Wrapper for."""

    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csc_matrix")

    if start_node_ids is None:
        start_node_ids = np.arange(A.shape[0])

    # Extract the information on the csr matrix
    logger.debug(
        "simulate random walks: network of {} nodes. {} walkers per node. {} start nodes".format(
            A.shape[0], num_walk, len(start_node_ids)
        )
    )
    if is_cum_trans_prob_mat is False:
        logger.debug("Calculating the transition probability")
        P = calc_cum_trans_prob(A)
    else:
        P = A

    logger.debug("Start simulation")
    walks = []
    sz = 0
    while sz <= (walk_length * num_walk * A.shape[0]):
        _walks, n_sampled_walks = _simulate_non_backtracking_walk(
            P.indptr,
            P.indices,
            P.data.astype(float),
            np.repeat(start_node_ids, num_walk),
            walk_length,
        )
        sz += n_sampled_walks
        walks.append(_walks)
    if len(walks) > 1:
        walks = np.vstack(walks)
    else:
        walks = walks[0]
    return walks


@njit(nogil=True)
def _simulate_non_backtracking_walk(
    A_indptr, A_indices, A_data, start_node_ids, walk_length,  # should be cumulative
):
    """Sampler based on a non-backtracking walk. A random walker chooses a
    neighbor with probability proportional to edge weights. For a fast
    simulation of random walks, we exploit scipy.sparse.csr_matrix data
    structure. In scipy.sparse.csr_matrix A, there are 3 attributes:

    - A.indices : column ID
    - A.data : element value
    - A.indptr : the first index of A.indices for the ith row
    The neighbors of node i are given by A.indices[A.indptr[i]:A.indptr[i+1]], and
    the edge weights are given by A.data[A.indptr[i]:A.indptr[i+1]].
    """
    # Alocate a memory for recording a walk
    walks = -np.ones((len(start_node_ids), walk_length), dtype=np.int32)
    n_sampled_walks = 0
    for sample_id in prange(len(start_node_ids)):
        start_node = start_node_ids[sample_id]
        # Record the starting node
        visit = start_node
        prev_visit = -1
        walks[sample_id, 0] = visit
        n_sampled_walks += 1
        for t in range(1, walk_length):
            # Compute the number of neighbors
            outdeg = A_indptr[visit + 1] - A_indptr[visit]
            neighbors = A_indices[A_indptr[visit] : A_indptr[visit + 1]]
            # If reaches to an absorbing state, finish the walk
            if outdeg == 0:
                break
            elif (outdeg == 1) & (neighbors[0] == prev_visit):
                break
            else:
                # find a neighbor by a roulette selection
                next_node = prev_visit
                while next_node == prev_visit:
                    _next_node = np.searchsorted(
                        A_data[A_indptr[visit] : A_indptr[visit + 1]],
                        np.random.rand(),
                        side="right",
                    )
                    next_node = neighbors[_next_node]
            # Record the transition
            walks[sample_id, t] = next_node
            # Move
            prev_visit = visit
            visit = next_node
            n_sampled_walks += 1
    return walks, n_sampled_walks


#
# Helper function
#
def calc_cum_trans_prob(A):
    P = A.copy()
    a = _calc_cum_trans_prob(P.indptr, P.indices, P.data.astype(float), P.shape[0])
    P.data = a
    return P


@njit(nogil=True)
def _calc_cum_trans_prob(
    A_indptr, A_indices, A_data_, num_nodes,  # should be cumulative
):
    A_data = A_data_.copy()
    for i in range(num_nodes):
        # Compute the out-deg
        outdeg = np.sum(A_data[A_indptr[i] : A_indptr[i + 1]])
        A_data[A_indptr[i] : A_indptr[i + 1]] = np.cumsum(
            A_data[A_indptr[i] : A_indptr[i + 1]]
        ) / np.maximum(outdeg, 1)
    return A_data


@njit(nogil=True)
def sample_columns_from_cum_prob(rows, A_indptr, A_indices, A_data):
    retvals = -np.ones(len(rows))
    for i in range(len(rows)):
        r = rows[i]
        nnz_row = A_indptr[r + 1] - A_indptr[r]
        if nnz_row == 0:
            continue

        # find a neighbor by a roulette selection
        _ind = np.searchsorted(
            A_data[A_indptr[r] : A_indptr[r + 1]], np.random.rand(), side="right",
        )
        retvals[i] = A_indices[A_indptr[r] + _ind]
    return retvals


def simulate_simple_walk(
    A,
    num_walk,
    walk_length,
    restart_prob,
    start_node_ids=None,
    restart_at_dangling=False,
    is_cum_trans_prob_mat=False,
    random_teleport=False,
    **params
):
    """Wrapper for."""

    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csc_matrix")

    if start_node_ids is None:
        start_node_ids = np.arange(A.shape[0])

    # Extract the information on the csr matrix
    logger.debug(
        "simulate random walks: network of {} nodes. {} walkers per node. {} start nodes".format(
            A.shape[0], num_walk, len(start_node_ids)
        )
    )
    if is_cum_trans_prob_mat is False:
        logger.debug("Calculating the transition probability")
        P = calc_cum_trans_prob(A)
    else:
        P = A

    logger.debug("Start simulation")
    return _simulate_simple_walk(
        P.indptr,
        P.indices,
        P.data.astype(float),
        P.shape[0],
        float(restart_prob),
        np.repeat(start_node_ids, num_walk),
        walk_length,
        restart_at_dangling,
        random_teleport,
    )


# @numba.jit(nopython=True, parallel=True)
@njit(nogil=True)
def _simulate_simple_walk(
    A_indptr,
    A_indices,
    A_data,  # should be cumulative
    num_nodes,
    restart_prob,
    start_node_ids,
    walk_length,
    restart_at_dangling,
    random_teleport,
):
    """Sampler based on a simple walk. A random walker chooses a neighbor with
    proportional to edge weights. For a fast simulation of random walks, we
    exploit scipy.sparse.csr_matrix data structure. In scipy.sparse.csr_matrix
    A, there are 3 attributes:

    - A.indices : column ID
    - A.data : element value
    - A.indptr : the first index of A.indices for the ith row
    The neighbors of node i are given by A.indices[A.indptr[i]:A.indptr[i+1]], and
    the edge weights are given by A.data[A.indptr[i]:A.indptr[i+1]].
    Parameters
    ----------
    A: scipy.sparse.csr_matrix
        Adjacency matrix of a network
    walk_length: int
        Length of a single walk
    window_length: int
        Length of a window rolling over a generated sentence
    restart_prob: float (Optional; Default 0)
        Restart probability
    return_sentence bool (Optional; Default True)
        Return sentence in form of numpy.ndarray. Set false if using residual2vec
            Returns
    -------
    sampler: func
        A function that takes start node ids (numpy.ndarray) as input.
        If return_sentence is True, sampler returns sentences in form of numpy.ndarray
        where sentence[n,t] indicates the tth word in the nth sentence.
        If return_sentence is False, sampler returns center-context node pairs and its frequency:
        - center: numpy.ndarray
            Center words
        - context: numpy.ndarray
            Context words
        - freq: numpy.ndarray
            Frequency
    """
    # Alocate a memory for recording a walk
    walks = -np.ones((len(start_node_ids), walk_length), dtype=np.int32)
    for sample_id in prange(len(start_node_ids)):
        start_node = start_node_ids[sample_id]
        # Record the starting node
        visit = start_node
        walks[sample_id, 0] = visit
        for t in range(1, walk_length):
            # Compute the number of neighbors
            outdeg = A_indptr[visit + 1] - A_indptr[visit]
            # If reaches to an absorbing state, finish the walk
            # the random walker is teleported back to the starting node
            # or Random walk with restart
            if outdeg == 0:
                if restart_at_dangling:
                    if random_teleport:
                        next_node = np.random.randint(0, num_nodes)
                    else:
                        next_node = start_node
                else:
                    if t == 1:  # when starting from sink
                        pass
                        walks[sample_id, t] = visit
                    break
            elif np.random.rand() <= restart_prob:
                if random_teleport:
                    next_node = np.random.randint(0, num_nodes)
                else:
                    next_node = start_node
            else:
                # find a neighbor by a roulette selection
                _next_node = np.searchsorted(
                    A_data[A_indptr[visit] : A_indptr[visit + 1]],
                    np.random.rand(),
                    side="right",
                )
                next_node = A_indices[A_indptr[visit] + _next_node]
            # Record the transition
            walks[sample_id, t] = next_node
            # Move
            visit = next_node
    return walks


def sample_center_context_pair(
    A,
    num_walks,
    walk_length,
    restart_prob,
    window_length,
    random_teleport=False,
    batch_size=100000,
):

    num_nodes = A.shape[0]
    num_chunk = np.ceil(num_walks * num_nodes / batch_size).astype(int)
    pair_cnt = Counter()

    logger.debug("Calculating the transition probability")
    P = calc_cum_trans_prob(A)
    for ch_id, start_node_ids in enumerate(
        np.array_split(np.arange(num_nodes), num_chunk)
    ):
        logger.debug("chunk {} / {}: simulate random walk".format(ch_id, num_chunk))
        walks = simulate_simple_walk(
            P,
            num_walks,
            walk_length,
            restart_prob,
            start_node_ids=start_node_ids,
            is_cum_trans_prob_mat=True,
            random_teleport=random_teleport,
        )

        walk_length = walks.shape[1]
        walk_num = walks.shape[0]
        L_single = window_length * (walk_length - window_length)
        L_all = walk_num * L_single

        logger.debug(
            "chunk {} / {}: generating {} center context pairs".format(
                ch_id, num_chunk, L_all
            )
        )
        walks = walks.astype(int)
        pairs = _generate_center_context_pair_ids(walks, window_length)
        pair_cnt += Counter(pairs)
        logger.debug(
            "chunk {} / {}: {} pairs sampled in total".format(
                ch_id, num_chunk, len(pair_cnt)
            )
        )

    ids = np.fromiter(pair_cnt.keys(), dtype=int)
    freq = np.fromiter(pair_cnt.values(), dtype=float)
    w = np.floor((np.sqrt(8 * ids + 1) - 1) * 0.5)
    t = (w ** 2 + w) * 0.5
    context = ids - t
    center = w - context
    return center.astype(int), context.astype(int), freq


@njit(nogil=True)
def _generate_center_context_pair_ids(walks, window_length):

    """Generate center context node pairs from walks."""
    #
    # Allocate a memory for center and context node
    #
    walk_length = walks.shape[1]
    walk_num = walks.shape[0]

    L_single = window_length * (walk_length - window_length)
    L_all = walk_num * L_single
    pairs = -np.ones(L_all)

    #
    # Simulate the random walk
    #
    for sample_id in prange(walk_num):
        # Tie center and context nodes
        # Two node ids are converted to a single int id using the Canter pairng
        for t in range(L_single):
            t0, t1 = divmod(t, window_length)
            pid = sample_id * L_single + t
            t1 = t0 + t1 + 1
            if (walks[sample_id, t0] < 0) or (walks[sample_id, t1] < 0):
                continue
            # Cantor pairing function
            pairs[pid] = int(walks[sample_id, t0]) + int(walks[sample_id, t1])  #
            pairs[pid] = pairs[pid] * (pairs[pid] + 1) / 2 + int(walks[sample_id, t1])
    pairs = pairs[pairs >= 0]
    return pairs


def generate_center_context_pair(walks, window_length):
    return _generate_center_context_pair(walks.astype(np.int64), int(window_length))


@njit(nogil=True)
def _generate_center_context_pair(walks, window_length):
    """Generate center context node pairs from walks."""

    #
    # Allocate a memory for center and context node
    #
    walk_length = walks.shape[1]
    walk_num = walks.shape[0]

    L_single = window_length * (walk_length - window_length)
    L_all = walk_num * L_single
    pairs = -np.ones(L_all)

    #
    # Simulate the random walk
    #
    for sample_id in prange(walk_num):

        # Tie center and context nodes
        # Two node ids are converted to a single int id using the Canter pairng
        for t in range(L_single):
            t0, t1 = divmod(t, window_length)
            pid = sample_id * L_single + t
            t1 = t0 + t1 + 1
            if (walks[sample_id, t0] < 0) or (walks[sample_id, t1] < 0):
                continue
            # Cantor pairing function
            pairs[pid] = int(walks[sample_id, t0]) + int(walks[sample_id, t1])  #
            pairs[pid] = pairs[pid] * (pairs[pid] + 1) / 2 + int(walks[sample_id, t1])
    pairs = pairs[pairs >= 0]

    # Count center-context pairs
    freq = np.bincount(pairs.astype(np.int64))
    ids = np.nonzero(freq)[0]
    freq = freq[ids]

    # Deparing
    w = np.floor((np.sqrt(8 * ids + 1) - 1) * 0.5)
    t = (w ** 2 + w) * 0.5
    context = ids - t
    center = w - context
    return center, context, freq
