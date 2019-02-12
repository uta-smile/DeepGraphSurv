import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.spatial.distance
import matplotlib.pyplot as plt


def compute_adj_matrix(adj_lists):
    """convert adj_list of graph to adjacency matrix"""

    neighbors = {}  # atom_id : list of neighbor atom
    for index, list in enumerate(adj_lists):
        neighbors[index] = list  # atom i and its adj list
    adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(neighbors))
    return adj_matrix


def get_adjacency(points):
    n_p = points.shape[0]

    # pre-learn the distance matrix, use mean as threshold to generate adjacency matrix, i.e. who connect who
    all_dist = []
    for i in range(n_p):
        for j in range(i + 1):
            all_dist += [np.linalg.norm(points[i] - points[j])]
            # input points are new points after clustering
    sparse_ratio = 0.3  # this ratio is to the percentage of points in matrix that finally has non-zero values
    cut_off_idx = int(len(all_dist) * sparse_ratio)
    d_lim = np.sort(all_dist)[cut_off_idx]

    adj_matrix = np.zeros((n_p, n_p))
    adj_list = [[] for _ in range(n_p)]
    for i in range(n_p):
        for j in range(i + 1, n_p):
            if np.linalg.norm(points[i] - points[j]) < d_lim:
                adj_matrix[i][j] = True
                adj_matrix[j][i] = True

                # adjacency list has redundancy
                adj_list[i].append(j)
                adj_list[j].append(i)

    return adj_list, adj_matrix


def compute_laplacian(adj_lists, normalized=True):
    def laplacian(adj_m, normalized=True):
        """Return the Laplacian of the weigth matrix.
        input: adjacency matrix (usually normalized)
        """
        W = adj_m
        # Degree matrix.
        d = np.sum(W, axis=0)

        # Laplacian matrix.
        if not normalized:
            D = np.diag(np.squeeze(d))
            L = D - W
        else:
            d += np.spacing(np.array(0, W.dtype))
            d = np.power(d.squeeze(), -0.5).flatten()
            D = np.diag(np.squeeze(d))
            I = np.identity(D.shape[0], dtype=W.dtype)
            L = I - D * W * D

        assert np.abs(L - L.T).mean() < 1e-9    # symmetric
        # assert type(L) is sp.csr_matrix
        return L

    def normalize_adj(adj_m):
        """Symmetrically normalize adjacency matrix.
        adj_m -> dense adjacency matrix of graph
        """
        adj = sp.coo_matrix(adj_m)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return normalized_adj.toarray()

    def preprocess_adj(adj_m):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        normalized_adj_m = normalize_adj(adj_m + sp.eye(adj_m.shape[0]))
        return normalized_adj_m  # return sparse matrix -> normalized  adj_matrix

    adj_m = compute_adj_matrix(adj_lists).astype(np.float32)
    adj_m = preprocess_adj(adj_m)
    lap_m = laplacian(adj_m, normalized=normalized)  # normalized is True, will give normalized Laplacian matrix
    return lap_m.astype(np.float32)


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return sp.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]


def fourier(L, algo='eigh', k=1):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U


def plot_spectrum(L, algo='eig'):
    """Plot the spectrum of a list of multi-scale Laplacians L."""
    # Algo is eig to be sure to get all eigenvalues.
    plt.figure(figsize=(17, 5))
    for i, lap in enumerate(L):
        lamb, U = fourier(lap, algo)
        step = 2**i
        x = range(step//2, L[0].shape[0], step)
        lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
        plt.plot(x, lamb, '.', label=lb)
    plt.legend(loc='best')
    plt.xlim(0, L[0].shape[0])
    plt.ylim(ymin=0)


def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN)."""
    M, N = X.shape
    assert L.dtype == X.dtype

    # L = rescale_L(L, lmax)
    # Xt = T @ X: MxM @ MxN.
    Xt = np.empty((K, M, N), L.dtype)
    # Xt_0 = T_0 X = I X = X.
    Xt[0, ...] = X
    # Xt_1 = T_1 X = L X.
    if K > 1:
        Xt[1, ...] = L.dot(X)
    # Xt_k = 2 L Xt_k-1 - Xt_k-2.
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
    return Xt


def lanczos(L, X, K):
    """
    Given the graph Laplacian and a data matrix, return a data matrix which can
    be multiplied by the filter coefficients to filter X using the Lanczos
    polynomial approximation.
    """
    M, N = X.shape
    assert L.dtype == X.dtype

    def basis(L, X, K):
        """
        Lanczos algorithm which computes the orthogonal matrix V and the
        tri-diagonal matrix H.
        """
        a = np.empty((K, N), L.dtype)
        b = np.zeros((K, N), L.dtype)
        V = np.empty((K, M, N), L.dtype)
        V[0, ...] = X / np.linalg.norm(X, axis=0)
        for k in range(K-1):
            W = L.dot(V[k, ...])
            a[k, :] = np.sum(W * V[k, ...], axis=0)
            W = W - a[k, :] * V[k, ...] - (
                    b[k, :] * V[k-1, ...] if k > 0 else 0)
            b[k+1, :] = np.linalg.norm(W, axis=0)
            V[k+1, ...] = W / b[k+1, :]
        a[K-1, :] = np.sum(L.dot(V[K-1, ...]) * V[K-1, ...], axis=0)
        return V, a, b

    def diag_H(a, b, K):
        """Diagonalize the tri-diagonal H matrix."""
        H = np.zeros((K*K, N), a.dtype)
        H[:K**2:K+1, :] = a
        H[1:(K-1)*K:K+1, :] = b[1:, :]
        H.shape = (K, K, N)
        Q = np.linalg.eigh(H.T, UPLO='L')[1]
        Q = np.swapaxes(Q, 1, 2).T
        return Q

    V, a, b = basis(L, X, K)
    Q = diag_H(a, b, K)
    Xt = np.empty((K, M, N), L.dtype)
    for n in range(N):
        Xt[..., n] = Q[..., n].T.dot(V[..., n])
    Xt *= Q[0, :, np.newaxis, :]
    Xt *= np.linalg.norm(X, axis=0)
    return Xt  # Q[0, ...]