import numpy as np
from numba import njit, prange


# ---------------------------------------------------------
# 1. The Low-Level Logic (Numba Optimized)
# ---------------------------------------------------------

@njit(parallel=True)
def canonicalize_points(data, p, q):
    """
    In-place sorts the p-part and q-part of every row.
    data: Shape (N, p + q + 1). The last column is the set_id.
    """
    n_rows = data.shape[0]

    # Parallel loop over all points
    for i in prange(n_rows):
        # 1. Sort the first p elements (Bubble sort is fine for small p)
        # Or use explicit slicing if p/q are fixed small constants for speed

        # Sort x_1...x_p
        # (Using a simple bubble/insertion sort here because
        #  overhead of calling np.sort on tiny arrays inside a loop can be high)
        for j in range(p):
            for k in range(p - 1):
                if data[i, k] > data[i, k + 1]:
                    temp = data[i, k]
                    data[i, k] = data[i, k + 1]
                    data[i, k + 1] = temp

        # Sort x'_1...x'_q (starts at index p)
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
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])  # Path compression
    return parent[i]


# ---------------------------------------------------------
# 2. The High-Level Driver
# ---------------------------------------------------------

def solve_grouping(list_of_sets, p, q):
    """
    list_of_sets: List of numpy arrays, where each array is a Set of points.
    """

    # A. PREPARE DATA
    # Flatten everything into one big matrix: [ coords... | set_id ]
    # This is standard numpy manipulation, usually fast enough without Numba
    all_rows = []
    total_sets = len(list_of_sets)

    for set_id, points in enumerate(list_of_sets):
        # Attach set_id to every point
        # Assuming points is (N, p+q)
        n = points.shape[0]
        ids = np.full((n, 1), set_id)
        combined = np.hstack((points, ids))
        all_rows.append(combined)

    # Create the giant matrix
    big_matrix = np.vstack(all_rows)

    # B. PARALLEL CANONICALIZE
    # This sorts the p and q sub-parts of every row in parallel
    canonicalize_points(big_matrix, p, q)

    # C. GLOBAL SORT
    # Sort the rows lexicographically so identical points end up adjacent.
    # We sort by the coordinates (columns 0 to p+q)
    # np.lexsort sorts by columns in reverse order, so we pass them reversed
    # or use a structured array / void view for sorting.
    # The easiest way for simple float/int arrays:

    # View as void to sort rows as blocks
    view = np.ascontiguousarray(big_matrix[:, :p + q]).view(np.dtype((np.void, big_matrix.dtype.itemsize * (p + q))))
    order = np.argsort(view.ravel())  # Get sort indices
    sorted_matrix = big_matrix[order]

    # D. SCAN AND LINK
    uf_parent = np.arange(total_sets)
    scan_and_link(sorted_matrix, p, q, uf_parent)

    return uf_parent