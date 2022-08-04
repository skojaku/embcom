import numba
import numpy as np
from scipy import sparse
import faiss
from sklearn.linear_model import LogisticRegression


class Louvain4Vectors:
    def __init__(self, num_neighbors=100, iteration=50, p0=None, device="cuda:1"):
        self.num_neighbors = num_neighbors
        self.iteration = iteration
        self.p0 = p0
        self.device = device

    def clustering(self, Z, return_member_matrix=False):
        """A Louvain algorithm for the Potts model."""
        if Z.shape[0] == 1:
            cids = np.array([0])
            if return_member_matrix:
                return sparse.csr_matrix(cids)
            return cids

        if self.p0 is None:
            self.fit(Z)

        num_nodes = Z.shape[0]
        node_size = np.ones(num_nodes)

        U = sparse.identity(num_nodes, format="csr")

        Vt = Z.copy()
        while True:
            cids_t = label_switching(
                Z=Vt,
                num_neighbors=self.num_neighbors,
                rho=self.p0,
                node_size=node_size,
                epochs=self.iteration,
                device=self.device,
            )
            _, cids_t = np.unique(cids_t, return_inverse=True)
            if int(max(cids_t) + 1) == Vt.shape[0]:
                break

            Ut = toMemberMatrix(cids_t)

            U = U @ Ut
            Vt = Ut.T @ Vt
            node_size = np.array(Ut.T @ node_size).reshape(-1)

        if return_member_matrix:
            return U

        cids = np.array((U @ sparse.diags(np.arange(U.shape[1]))).sum(axis=1)).reshape(
            -1
        )
        return cids

    def fit(self, emb):
        rpos, cpos, vpos = find_knn_edges(
            emb, num_neighbors=self.num_neighbors, device=self.device
        )
        cneg = np.random.choice(emb.shape[0], len(cpos))
        vneg = np.array(np.sum(emb[rpos, :] * emb[cneg, :], axis=1)).reshape(-1)

        model = LogisticRegression()
        model.fit(
            np.concatenate([vpos, vneg]).reshape((-1, 1)),
            np.concatenate([np.ones_like(vpos), np.zeros_like(vneg)]),
        )
        self.w1, self.b0 = model.coef_[0, 0], -model.intercept_[0]
        self.p0 = self.b0 / self.w1


#
# Clustering based on a label switching algorithm
#
def label_switching(Z, rho, num_neighbors=50, node_size=None, device=None, epochs=50):

    num_nodes, dim = Z.shape

    if node_size is None:
        node_size = np.ones(num_nodes)

    Z = Z.copy(order="C").astype(np.float32)

    # Construct the candidate graph
    Z1 = np.hstack([Z, np.ones((num_nodes, 1))])
    Zrho = np.hstack([Z, -rho * node_size.reshape((-1, 1))])

    r, c, v = find_knn_edges(
        Zrho,
        target=Z1,
        num_neighbors=num_neighbors,
        metric="cosine",
        sim_threshold=0,
        device=device,
    )
    A = sparse.csr_matrix((v, (r, c)), shape=(num_nodes, num_nodes))
    return _label_switching_(
        A_indptr=A.indptr,
        A_indices=A.indices,
        Z=Z,
        num_nodes=num_nodes,
        rho=rho,
        node_size=node_size,
        epochs=epochs,
    )


@numba.jit(nopython=True, cache=True)
def _label_switching_(A_indptr, A_indices, Z, num_nodes, rho, node_size, epochs=100):

    Nc = np.zeros(num_nodes)
    cids = np.arange(num_nodes)
    Vc = Z.copy()
    Vnorm = np.sum(np.multiply(Z, Z), axis=1).reshape(-1)
    for nid in range(num_nodes):
        Nc[nid] += node_size[nid]

    for _it in range(epochs):
        order = np.random.choice(num_nodes, size=num_nodes, replace=False)
        updated_node_num = 0

        for _k, node_id in enumerate(order):

            # Get the weight and normalized weight
            neighbors = A_indices[A_indptr[node_id] : A_indptr[node_id + 1]]

            # Calculate the grain
            c = cids[node_id]
            clist = np.unique(cids[neighbors])
            next_cid = -1
            dqmax = 0
            qself = (
                np.sum(Z[node_id, :] * Vc[c, :])
                - Vnorm[node_id]
                - rho * node_size[node_id] * (Nc[c] - node_size[node_id])
            )
            for cprime in clist:
                if c == cprime:
                    continue
                dq = (
                    np.sum(Z[node_id, :] * Vc[cprime, :])
                    - rho * node_size[node_id] * Nc[cprime]
                ) - qself
                if dqmax < dq:
                    next_cid = cprime
                    dqmax = dq

            if dqmax <= 1e-16:
                continue

            Nc[c] -= node_size[node_id]
            Nc[next_cid] += node_size[node_id]

            Vc[c, :] -= Z[node_id, :]
            Vc[next_cid, :] += Z[node_id, :]

            cids[node_id] = next_cid
            updated_node_num += 1

        if (updated_node_num / np.maximum(1, num_nodes)) < 1e-3:
            break
    return cids


# Evaluate the clustering score
def eval(V, U, rho, node_size=None):

    num_nodes = V.shape[0]

    if not isinstance(U, sparse.csr_matrix):
        U = toMemberMatrix(U)

    if node_size is None:
        node_size = np.ones(num_nodes)

    UV = U.T @ V
    W = UV @ UV.T
    Ns = np.array(U.T @ node_size).reshape(-1)
    Vsqrt = np.sum(np.linalg.norm(V, axis=1) ** 2)
    return np.sum(np.diag(W)) - Vsqrt - rho * np.sum(Ns * (Ns - 1))


#
# Helper
#
def toMemberMatrix(group_ids):
    n = len(group_ids)
    k = int(np.max(group_ids) + 1)
    return sparse.csr_matrix((np.ones(n), (np.arange(n), group_ids)), shape=(n, k))


def find_knn_edges(
    emb,
    num_neighbors,
    sim_threshold=0,
    target=None,
    metric="cosine",
    device=None,
):

    k = int(np.minimum(num_neighbors + 1, emb.shape[0]).astype(int))
    indices, distances = find_knn(
        emb if target is None else target,
        emb,
        num_neighbors=k,
        metric=metric,
        device=device,
    )

    r = np.outer(np.arange(indices.shape[0]), np.ones((1, indices.shape[1]))).astype(
        int
    )
    r, c, distances = (
        r.reshape(-1),
        indices.astype(int).reshape(-1),
        distances.reshape(-1),
    )

    if metric == "cosine":
        s = (r != c) & (distances >= sim_threshold)
    else:
        s = (r != c) & (distances <= sim_threshold)

    r, c, distances = r[s], c[s], distances[s]

    if len(r) == 0:
        return r, c, distances

    return r, c, distances


def find_knn(target, emb, num_neighbors, metric="cosine", device=None):
    if metric == "cosine":
        index = faiss.IndexFlatIP(emb.shape[1])
    else:
        index = faiss.IndexFlatL2(emb.shape[1])
    if device is None:
        index.add(emb.astype(np.float32))
        distances, indices = index.search(target.astype(np.float32), k=num_neighbors)
    else:
        try:
            gpu_id = int(device[-1])
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            index.add(emb.astype(np.float32))
            distances, indices = index.search(
                target.astype(np.float32), k=num_neighbors
            )
        except RuntimeError:
            if metric == "cosine":
                index = faiss.IndexFlatIP(emb.shape[1])
            else:
                index = faiss.IndexFlatL2(emb.shape[1])
            index.add(emb.astype(np.float32))
            distances, indices = index.search(
                target.astype(np.float32), k=num_neighbors
            )
    return indices, distances
