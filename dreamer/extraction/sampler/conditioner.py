import numpy as np
import scipy.optimize as opt
import sympy as sp
from fpylll import IntegerMatrix, LLL, BKZ


class Stage1_Conditioner:
    """
    Conditions a high-dimensional constrained space by finding the integer
    nullspace and applying LLL/BKZ lattice reduction to minimize basis skewness.
    """
    def __init__(self, A_prime, max_beta=10, defect_tolerance=5.0, tol=1e-9):
        self.A_prime = np.array(A_prime, dtype=np.float64)
        self.d_orig = self.A_prime.shape[1]
        self.max_beta = max_beta
        self.defect_tolerance = defect_tolerance
        self.tol = tol

    def process(self):
        """Main orchestrator: returns the conditioned basis and transformed bounds."""
        # print("[Stage 1] Extracting hyperplanes...")
        E, B_orig = self._extract_constraints()

        # print(f"[Stage 1] Computing raw integer nullspace (Equalities: {len(E)})...")
        Z_raw = self._compute_integer_basis(E)

        if Z_raw.shape[1] == 0:
            raise ValueError("The equality constraints result in a 0-dimensional space.")

        # print(f"[Stage 1] Flatland Dimension: {Z_raw.shape[1]}D. Initiating Reduction Ratchet...")
        Z_reduced, U_transform = self._ratchet_lattice_reduction(Z_raw)

        # print("[Stage 1] Transforming inequality bounds to new conditioned space...")
        B_reduced = self._transform_bounds(B_orig, Z_raw, U_transform)

        return Z_reduced, B_reduced, U_transform

    def _extract_constraints(self):
        """Separates A_prime into Equality (E) and Inequality (B) matrices."""
        eq_rows, ineq_rows = [], []
        m = self.A_prime.shape[0]

        for i in range(m):
            c = -self.A_prime[i]
            res = opt.linprog(c, A_ub=-self.A_prime, b_ub=np.zeros(m),
                              bounds=(-1, 1), method='highs')
            if res.success and -res.fun < 1e-7:
                eq_rows.append(self.A_prime[i])
            else:
                ineq_rows.append(self.A_prime[i])

        E = np.array(eq_rows, dtype=np.float64) if eq_rows else np.empty((0, self.d_orig))
        B = np.array(ineq_rows, dtype=np.float64) if ineq_rows else np.empty((0, self.d_orig))
        return E, B

    def _compute_integer_basis(self, E):
        """Finds the gapless integer basis for the equality hyperplanes."""
        if len(E) == 0:
            return np.eye(self.d_orig, dtype=np.int64)

        sp_matrix = sp.Matrix(E).applyfunc(sp.nsimplify)
        null_basis = sp_matrix.nullspace()

        if not null_basis:
            return np.zeros((self.d_orig, 0), dtype=np.int64)

        int_basis = []
        for vec in null_basis:
            common_denom = sp.Integer(1)
            for val in vec:
                common_denom = sp.lcm(common_denom, sp.Rational(val).q)
            int_basis.append(np.array(vec * common_denom, dtype=np.int64).flatten())

        return np.column_stack(int_basis)

    def _calculate_defect(self, Z):
        """
        Calculates the Orthogonality Defect.
        1.0 is a perfect hypercube. >100 is a highly distorted "pancake".
        """
        # Z columns are the basis vectors
        norms = np.linalg.norm(Z, axis=0)
        prod_norms = np.prod(norms)

        # Volume of the fundamental parallelepiped
        det_L = np.sqrt(np.abs(np.linalg.det(Z.T @ Z)))
        if det_L < 1e-9: return float('inf')

        return prod_norms / det_L

    def _ratchet_lattice_reduction(self, Z_int):
        """Dynamically applies LLL and BKZ via fpylll to orthogonalize the space."""
        # Note: fpylll uses row-matrices, so we transpose Z_int
        M_fpylll = IntegerMatrix.from_matrix(Z_int.T.tolist())
        U_fpylll = IntegerMatrix.identity(M_fpylll.nrows)

        # 1. Baseline: Standard LLL
        LLL.reduction(M_fpylll, U_fpylll)
        Z_current = np.array([list(row) for row in M_fpylll]).T
        defect = self._calculate_defect(Z_current)
        # print(f"  -> LLL applied. Orthogonality Defect: {defect:.2f}")

        # 2. Escalation Ratchet: BKZ
        beta = 4
        while defect > self.defect_tolerance and beta <= self.max_beta:
            # print(f"  -> Defect too high. Escalating to BKZ (Block Size: {beta})...")
            param = BKZ.Param(block_size=beta, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
            BKZ.reduction(M_fpylll, param, U=U_fpylll)

            Z_current = np.array([list(row) for row in M_fpylll]).T
            defect = self._calculate_defect(Z_current)
            # print(f"  -> BKZ-{beta} applied. New Defect: {defect:.2f}")
            beta += 2

        U_np = np.array([list(row) for row in U_fpylll])
        return Z_current, U_np

    def _transform_bounds(self, B_orig, Z_raw, U_np):
        """Applies the fpylll transformation matrix U to the inequality bounds."""
        if len(B_orig) == 0:
            return np.empty((0, Z_raw.shape[1]))

        # In fpylll: M_new = U * M_old.
        # Since M represents Z^T, this means Z_new = Z_old * U^T.
        # So our new flatland constraints are B * Z_old * U^T
        B_flat_raw = B_orig @ Z_raw
        B_reduced = B_flat_raw @ U_np.T
        return B_reduced