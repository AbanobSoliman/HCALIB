# Generated with SMOP  0.41-beta
import numpy as np

from libsmop import *


# bspline.m

def cumul_b_splineR3(P, u, N):
    S = np.zeros((3, len(u) * (P.shape[1] - N)))
    k = N + 1
    U = np.zeros((k, len(u)))
    dP = get_inc(P)
    M = calcM(k)
    for i in np.arange(1, k + 1).reshape(-1):
        U[i-1, :] = u ** (i - 1)

    zz = 1
    for i in np.arange(1, P.shape[2 - 1] - k + 1 + 1).reshape(-1):
        B = np.concatenate((np.array([P[:, i-1]]).T, dP[:, np.arange(i, i + k - 2 + 1)-1]), axis=1) @ M @ U
        S[:, np.arange(zz, zz + B.shape[2 - 1] - 1 + 1)-1] = B
        zz = zz + B.shape[2 - 1]

    return S


def get_inc(X):
    dX = np.zeros((X.shape[1 - 1], X.shape[2 - 1] - 1))
    for i in np.arange(0, X.shape[2 - 1] - 1).reshape(-1):
        dX[:, i] = X[:, i + 1] - X[:, i]
    return dX


def calcM(k):
    M = np.zeros((k, k))
    m = np.zeros((k, k))
    s = np.arange(0, k - 1 + 1, 1)
    n = np.arange(0, k - 1 + 1, 1)
    L = np.arange(0, k - 1 + 1, 1)
    for i in np.arange(1, k + 1).reshape(-1):
        for j in np.arange(1, k + 1).reshape(-1):
            add = 0.0
            for l in np.arange(i, k + 1).reshape(-1):
                add = add + float(
                    (- 1) ** (L[l-1] - s[i-1]) * nchoosek(k, L[l-1] - s[i-1]) * (k - 1 - L[l-1]) ** (k - 1 - n[j-1]))
            m[i-1, j-1] = (nchoosek(k - 1, n[j-1]) / (fact(k - 1))) * add

    for j in np.arange(1, k + 1).reshape(-1):
        for n in np.arange(1, k + 1).reshape(-1):
            M[j-1, n-1] = sum(m[np.arange(j, k + 1)-1, n-1])

    return M


def fact(n):
    return 1 if (n == 1 or n == 0) else n * fact(n - 1)


def nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n / k * nchoosek(n - 1, k - 1)
    return round(r)
