from typing import Optional, Sequence, Tuple, Any

import pandas as pd
from numpy.typing import NDArray
import numpy as np
from scipy.sparse import coo_array, csr_array, diags, dok_array
from scipy.sparse.linalg import eigs

from scipy.constants import k as kB, N_A


def window(seq: Sequence, len_window: int, step: int = 1) -> Tuple[Any, Any]:
    """
    How this works: returns a list of sublists. Each sublist has two elements: [start_element, end_element].
    Both elements are taken from the seq and are len_window elements apart. The parameter step controls what
    the step between subsequent start_elements is.

    Yields:
        [start_element, end_element] where both elements com from seq where they are len_window positions
        apart

    Example:
        >>> gen_window = window([1, 2, 3, 4, 5, 6], len_window=2, step=3)
        >>> next(gen_window)
        (1, 3)
        >>> next(gen_window)
        (4, 6)
    """
    # in this case always move the window by step and use all points in simulations to count transitions
    for k in range(0, len(seq) - len_window, step):
        start_stop_list = seq[k: k + len_window + 1:len_window]
        if not np.isnan(start_stop_list).any():
            yield tuple([int(el) for el in start_stop_list if not np.isnan(el)])


def noncorr_window(seq: Sequence, len_window: int) -> Tuple[Any, Any]:
    """
    Subsample the seq so that only each len_window-th element remains and then similarly return pairs of
    elements.

    Example:
        >>> gen_obj = noncorr_window([1, 2, 3, 4, 5, 6, 7], 3)
        >>> next(gen_obj)
        (1, 4)
        >>> next(gen_obj)
        (4, 7)
    """
    # in this case, only use every len_window-th element for MSM. Faster but loses a lot of data
    return window(seq, len_window, step=len_window)


class MSM:

    """
    From assignments create a MSM transition matrix.
    """

    def __init__(self, assigned_trajectory: NDArray, total_num_cells: int):
        self.assigned_trajectory = assigned_trajectory
        self.total_num_cells = total_num_cells

    def get_one_tau_transition_matrix(self, tau: float, noncorrelated_windows: bool):
        sparse_count_matrix = dok_array((self.total_num_cells, self.total_num_cells))
        # save the number of transitions between cell with index i and cell with index j
        # count_per_cell = {(i, j): 0 for i in range(self.num_cells) for j in range(self.num_cells)}
        if noncorrelated_windows:
            window_cell = noncorr_window(self.assigned_trajectory, int(tau))
        else:
            window_cell = window(self.assigned_trajectory, int(tau))

        for cell_slice in window_cell:
            if len(cell_slice)<2:
                pass
            else:
                el1, el2 = cell_slice
                sparse_count_matrix[el1, el2] += 1
                # enforce detailed balance
                sparse_count_matrix[el2, el1] += 1
        sparse_count_matrix = sparse_count_matrix.tocsr()
        sums = sparse_count_matrix.sum(axis=1)
        # to avoid dividing by zero
        sums[sums == 0] = 1
        # now dividing with counts (actually multiplying with inverse)
        diagonal_values = np.reciprocal(sums)
        diagonal_matrix = diags(diagonal_values, format='csr')
        # Left multiply the CSR matrix with the diagonal matrix
        return diagonal_matrix.dot(sparse_count_matrix)

    def get_all_tau_transition_matrices(self, taus: NDArray, noncorrelated_windows: bool):
        transition_matrix = np.zeros(shape=taus.shape, dtype=object)
        for tau_i, tau in enumerate(taus):
            transition_matrix[tau_i] = self.get_one_tau_transition_matrix(tau, noncorrelated_windows=noncorrelated_windows)
        return transition_matrix



class SQRA:

    def __init__(self, energies: NDArray, volumes: NDArray, distances: csr_array, surfaces: csr_array):
        self.energies = energies
        self.volumes = volumes
        self.distances = distances
        self.surfaces = surfaces

    def get_rate_matrix(self, D: float, T: float) -> csr_array:
        # calculating rate matrix
        # for sqra demand that each energy corresponds to exactly one cell
        assert len(self.energies) == len(self.volumes), f"{len(self.energies)} != {len(self.volumes)}"
        # you cannot multiply or divide directly in a coo format
        transition_matrix = D * self.surfaces  #/ all_distances
        print("data shape", self.surfaces.data.shape, self.distances.data.shape)
        transition_matrix = transition_matrix.tocoo()
        transition_matrix.data /= self.distances.tocoo().data
        # Divide every row of transition_matrix with the corresponding volume
        transition_matrix.data /= self.volumes[transition_matrix.row]
        print("done volumes")
        # multiply with sqrt(pi_j/pi_i) = e**((V_i-V_j)*1000/(2*k_B*N_A*T))
        # gromacs uses kJ/mol as energy unit, boltzmann constant is J/K
        diff_energies = self.energies[transition_matrix.row] - self.energies[transition_matrix.col]
        # cannot allow more than 3 orders of magnitude difference
        print(f"Warning! {len(np.where(diff_energies > 5e2)[0])} pairs of cells have a very large difference in "
              f"energy, more than factor 500. This would lead to overflow, so these differences are capped to a "
              f"factor 500. This might be a sign of poor discretisation or just the case of L-J overlap.")
        diff_energies = np.where(diff_energies < 5e2, diff_energies, 5e2)

        print("DIFF", diff_energies.shape, self.volumes.shape, self.distances.shape, self.surfaces.shape)

        pi_exponent = np.round(diff_energies,14) * 1000 / (2 * kB * N_A * T)

        print(pd.DataFrame(pi_exponent).describe())

        transition_matrix.data *= np.exp(pi_exponent)
        # normalise rows
        sums = transition_matrix.sum(axis=1)
        sums = np.array(sums).squeeze()
        all_i = np.arange(len(self.volumes))
        diagonal_array = coo_array((-sums, (all_i, all_i)), shape=(len(all_i), len(all_i)))
        transition_matrix = transition_matrix.tocsr() + diagonal_array.tocsr()
        return transition_matrix


class DecompositionTool:

    def __init__(self, matrix_to_decompose):
        """

        Args:
            matrix_to_decompose (): either a single matrix or an array of matrices (for different taus) we want to
            decompose
        """
        self.matrix_to_decompose = matrix_to_decompose

    def get_decomposition(self, tol: float, maxiter: int, which: str, sigma: Optional[float], k=12):
        """
        The function for users - will decompose all matrices.

            tol ():
            maxiter ():
            which ():
            sigma ():

        Returns:

        """
        eigenval, eigenvec = eigs(self.matrix_to_decompose.T, k=k, tol=tol, maxiter=maxiter, which=which, sigma=sigma)
        # if imaginary eigenvectors or eigenvalues, raise error
        if not np.allclose(eigenvec.imag.max(), 0, rtol=1e-3, atol=1e-5) or not np.allclose(eigenval.imag.max(), 0,
                                                                                            rtol=1e-3, atol=1e-5):
            print(f"Complex values for eigenvectors and/or eigenvalues: {eigenvec}, {eigenval}")
        eigenvec = eigenvec.real
        eigenval = eigenval.real
        # sort eigenvectors according to their eigenvalues
        idx = eigenval.argsort()[::-1]
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:, idx]
        return eigenval, eigenvec