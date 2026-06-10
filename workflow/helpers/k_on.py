import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

def solve_mfpt(Q, B):

    n = Q.shape[0]

    A = Q.copy().tolil()

    rhs = -np.ones(n)

    for b in B:
        A[b,:] = 0
        A[b,b] = 1
        rhs[b] = 0

    tau = spsolve(A.tocsr(), rhs)

    return tau


def mfpt_A_to_B(Q, pi, A, B):

    tau = solve_mfpt(Q, B)

    piA = pi[A] / np.sum(pi[A])

    return np.sum(piA * tau[A])


def kon(Q, pi, A, B):

    tauAB = mfpt_A_to_B(Q, pi, A, B)

    return 1.0 / tauAB