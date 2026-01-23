import numpy as np
from numba import njit, prange
from typing import List


@njit(parallel=True)
def canonicalize_points(data: np.ndarray, p: int, q: int):
    """
    In-place sorts the p-part and q-part of every row.
    :param data: Shape (N, p + q + 1). The last column is the set_id.
    :param p: The number of p-coordinates.
    :param q: The number of q-coordinates.
    """
    n_rows = data.shape[0]

    # Parallel loop over all points
    for i in prange(n_rows):
        # sort first p and last q elements
        for j in range(p):
            for k in range(p - 1):
                if data[i, k] > data[i, k + 1]:
                    temp = data[i, k]
                    data[i, k] = data[i, k + 1]
                    data[i, k + 1] = temp

        start_q = p
        end_q = p + q
        for j in range(q):
            for k in range(start_q, end_q - 1):
                if data[i, k] > data[i, k + 1]:
                    temp = data[i, k]
                    data[i, k] = data[i, k + 1]
                    data[i, k + 1] = temp


@njit
def scan_and_link(sorted_data, p, q, uf_parent):
    """
    Scans the sorted array. If row i matches row i+1 (excluding set_id),
    union their sets.
    """
    n_rows = sorted_data.shape[0]
    total_cols = p + q

    for i in range(n_rows - 1):
        # Check if the coordinates match exactly
        match = True
        for k in range(total_cols):
            if sorted_data[i, k] != sorted_data[i + 1, k]:
                match = False
                break

        if match:
            # The points are identical (canonicalized).
            # Merge the sets they belong to.
            set_a = int(sorted_data[i, total_cols])
            set_b = int(sorted_data[i + 1, total_cols])

            # Simple Union-Find "Union" operation
            root_a = find(uf_parent, set_a)
            root_b = find(uf_parent, set_b)
            if root_a != root_b:
                uf_parent[root_a] = root_b


# Standard Union-Find 'Find' helper
@njit
def find(parent, i):
    """
    Union-Find Find operation.
    :param parent: parent array for describing the upside-down tree.
    :param i: The node to find the root of.
    :return: The root of the set containing i.
    """
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])
    return parent[i]


def solve_grouping(list_of_sets: List[np.ndarray], p: int, q: int):
    """
    Using a group of disjoint points find the unified sets using the pFq symetry property
    :param list_of_sets: List of numpy arrays, where each array is a Set of points.
    :param p: The number of p-coordinates.
    :param q: The number of q-coordinates.
    """

    # prepare data
    all_rows = []
    total_sets = len(list_of_sets)
    for set_id, points in enumerate(list_of_sets):
        n = points.shape[0]
        ids = np.full((n, 1), set_id)
        combined = np.hstack((points, ids))
        all_rows.append(combined)
    big_matrix = np.vstack(all_rows)

    canonicalize_points(big_matrix, p, q)

    # smart sorting
    view = np.ascontiguousarray(big_matrix[:, :p + q]).view(np.dtype((np.void, big_matrix.dtype.itemsize * (p + q))))
    order = np.argsort(view.ravel())  # Get sort indices
    sorted_matrix = big_matrix[order]

    # preform scan and link of the union find
    uf_parent = np.arange(total_sets)
    scan_and_link(sorted_matrix, p, q, uf_parent)
    return uf_parent
