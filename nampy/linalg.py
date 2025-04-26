from . import utils
from .base import array


def center_data(X: array) -> array:
    mean_m = array([[utils.mean(X[:, i]) for i in range(X.shape[1])]] * X.shape[0])
    return X - mean_m


def covariance_matrix(X: array) -> array:
    return (1 / (X.shape[0] - 1)) * (X.T @ X)


def explained_variance_ratio(eigenvalues: [float], k: int) -> float:
    if not eigenvalues:
        raise ValueError("Eigenvalues array cannot be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(eigenvalues):
        raise ValueError("k cannot be greater than number of eigenvalues")
    if sum(eigenvalues) == 0:
        raise ValueError("Sum of eigenvalues cannot be zero")

    eigenvalues.sort(reverse=True)
    return sum(eigenvalues[:k]) / sum(eigenvalues)


def _outer(u, v):
    n = u.shape[0]
    M = utils.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = u[i, 0] * v[j, 0]
    return M


def _apply_householder_reflection(R, v, k):
    n = R.shape[0]
    for j in range(k, n):
        dot = 0.0
        for i in range(k, n):
            dot += v[i - k, 0] * R[i, j]
        for i in range(k, n):
            R[i, j] -= 2 * v[i - k, 0] * dot


def _apply_householder_to_Q(Q, v, k):
    n = Q.shape[0]
    for i in range(n):
        dot = 0.0
        for j in range(k, n):
            dot += Q[i, j] * v[j - k, 0]
        for j in range(k, n):
            Q[i, j] -= 2 * dot * v[j - k, 0]


def _householder_qr(A):
    n = A.shape[0]
    Q = utils.eye(n)
    R = A.copy()

    for k in range(n - 1):
        n_sub = n - k
        x = utils.zeros(shape=(n_sub, 1))
        for i in range(n_sub):
            x[i, 0] = R[k + i, k]

        e1 = utils.zeros(shape=(n_sub, 1))
        e1[0, 0] = 1.0

        alpha = x.norm
        sign = 1.0 if x[0, 0] >= 0 else -1.0
        v = x + e1 * (sign * alpha)
        v = v / v.norm

        _apply_householder_reflection(R, v, k)
        _apply_householder_to_Q(Q, v, k)

    return Q, R


def qr_eigen_decomposition(A, max_iter=None, tol=1e-8):
    n = A.shape[0]
    if max_iter is None:
        max_iter = 10 * n

    Ak = A.copy()
    Q_total = utils.eye(n)

    for _ in range(max_iter):
        # Wilkinson shift
        mu = Ak[n - 1, n - 1]
        I = utils.eye(n)
        shifted = Ak - mu * I

        Q, R = _householder_qr(shifted)
        Ak1 = R @ Q + mu * I
        Q_total = Q_total @ Q

        off_diag = 0.0
        for i in range(1, n):
            for j in range(i):
                off_diag += abs(Ak1[i, j]) + abs(Ak1[j, i])
        if off_diag < tol:
            Ak = Ak1
            break
        Ak = Ak1

    eigenvalues = [Ak[i, i] for i in range(n)]
    eigen_pairs = [
        (eigenvalues[i], [Q_total[j, i] for j in range(n)]) for i in range(n)
    ]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)

    sorted_eigenvalues = [pair[0] for pair in eigen_pairs]
    sorted_eigenvectors = utils.zeros(shape=(n, n))
    for i in range(n):
        eigenvector = eigen_pairs[i][1]
        for j in range(n):
            sorted_eigenvectors[j, i] = eigenvector[j]

    return sorted_eigenvalues, sorted_eigenvectors
