import ctypes
import numpy as np
from numba import jit, prange


@jit(nopython=True)
def k_means_euclid_py(X, K, max_iters=10000):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, K, replace=False)]
    labels = np.zeros(n_samples, dtype=np.int64)

    for _ in prange(max_iters):
        for i in prange(n_samples):
            distances = np.zeros(K, dtype=np.float64)
            for j in prange(K):
                distances[j] = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
            labels[i] = np.argmin(distances)

        new_centroids = np.zeros((K, n_features), dtype=np.float64)
        counts = np.zeros(K, dtype=np.int64)

        for i in prange(n_samples):
            new_centroids[labels[i]] += X[i]
            counts[labels[i]] += 1

        for k in prange(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


@jit(nopython=True)
def k_means_manhattan_py(X, K, max_iters=10000):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, K, replace=False)]
    labels = np.zeros(n_samples, dtype=np.int64)

    for _ in prange(max_iters):
        for i in prange(n_samples):
            distances = np.zeros(K, dtype=np.float64)
            for j in prange(K):
                distances[j] = np.sum(np.abs(X[i] - centroids[j]))
            labels[i] = np.argmin(distances)

        new_centroids = np.zeros((K, n_features), dtype=np.float64)
        counts = np.zeros(K, dtype=np.int64)

        for i in prange(n_samples):
            new_centroids[labels[i]] += X[i]
            counts[labels[i]] += 1

        for k in prange(K):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


lib = ctypes.cdll.LoadLibrary('./k_means_c.so')

# Define the argument and return types
lib.k_means_euclid.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # data_ptr
    ctypes.c_int,  # n_samples
    ctypes.c_int,  # n_features
    ctypes.c_int,  # K
    ctypes.c_int,  # max_iters
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # labels_ptr
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")  # centroids_ptr
]
lib.k_means_euclid.restype = None

# Define the argument and return types
lib.k_means_manhattan.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # data_ptr
    ctypes.c_int,  # n_samples
    ctypes.c_int,  # n_features
    ctypes.c_int,  # K
    ctypes.c_int,  # max_iters
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # labels_ptr
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")  # centroids_ptr
]
lib.k_means_manhattan.restype = None


def k_means_euclid_c(X, K, max_iters=10000):
    n_samples, n_features = X.shape
    X = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int32)
    centroids = np.zeros((K, n_features), dtype=np.float64)

    lib.k_means_euclid(X, n_samples, n_features, K, max_iters, labels, centroids)
    return labels, centroids


def k_means_manhattan_c(X, K, max_iters=10000):
    n_samples, n_features = X.shape
    X = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int32)
    centroids = np.zeros((K, n_features), dtype=np.float64)

    lib.k_means_manhattan(X, n_samples, n_features, K, max_iters, labels, centroids)
    return labels, centroids


def generate_data1(n_samples):
    X = np.random.rand(n_samples, 2) * 10
    Y = np.zeros(n_samples, dtype=np.int8) - 1
    for i in range(n_samples):
        if 0 < X[i, 0] < 3 and 0 < X[i, 1] < 3:
            Y[i] = 0
        elif 0 < X[i, 0] < 3 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 1
        elif 0 < X[i, 0] < 3 and 7 < X[i, 1] < 10:
            Y[i] = 2
        elif 3.5 < X[i, 0] < 6.5 and 0 < X[i, 1] < 3:
            Y[i] = 3
        elif 3.5 < X[i, 0] < 6.5 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 4
        elif 3.5 < X[i, 0] < 6.5 and 7 < X[i, 1] < 10:
            Y[i] = 5
        elif 7 < X[i, 0] < 10 and 0 < X[i, 1] < 3:
            Y[i] = 6
        elif 7 < X[i, 0] < 10 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 7
        elif 7 < X[i, 0] < 10 and 7 < X[i, 1] < 10:
            Y[i] = 8
    valid = Y >= 0
    X = X[valid]
    Y = Y[valid]
    return X, Y


def generate_data2(n_samples):
    X = np.random.rand(n_samples, 2) * 10
    Y = np.zeros(n_samples, dtype=np.int8)
    for i in range(n_samples):
        if 0 < X[i, 0] < 6 and 0 < X[i, 1] < 6:
            Y[i] = 1
        elif 7 < X[i, 0] < 10 and 0 < X[i, 1] < 3:
            Y[i] = 2
        elif 7 < X[i, 0] < 10 and 3 < X[i, 1] < 6:
            Y[i] = 3
        elif 0 < X[i, 0] < 3 and 7 < X[i, 1] < 10:
            Y[i] = 4
        elif 3 < X[i, 0] < 6 and 7 < X[i, 1] < 10:
            Y[i] = 5
        elif 7 < X[i, 0] < 10 and 7 < X[i, 1] < 10:
            Y[i] = 6
    valid = Y > 0
    X = X[valid]
    Y = Y[valid]
    return X, Y
