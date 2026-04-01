import time
import numpy as np
import matplotlib.pyplot as plt
from dreamer.extraction.sampler.e2e import EndToEndSamplingEngine


class TestHarness:
    """
    The definitive evaluation framework for 15D Lattice Sampling Engines.
    """
    def __init__(self, engine_class):
        self.EngineClass = engine_class
        self.results = {}

    def _generate_gauntlet_matrix(self, scenario_type, seeding=False):
        """
        Dynamically generates guaranteed-valid matrices for the 3 archetypes.
        """
        if scenario_type == "10D_Fat_Baseline":
            d_orig = 10
            num_eq, num_ineq = 3, 8
            coeff_range, thinness = 5, 1
        elif scenario_type == "15D_Needle":
            d_orig = 15
            num_eq, num_ineq = 4, 12
            coeff_range, thinness = 5, 10  # High thinness aggressively restricts the bounds
        elif scenario_type == "15D_Pancake":
            d_orig = 15
            num_eq, num_ineq = 4, 10
            coeff_range, thinness = 5, 1
        elif scenario_type == "15D_Realistic_Dense":
            d_orig = 15
            num_eq, num_ineq = 4, 80
            coeff_range, thinness = 5, 1

            # 1. Generate E and find its exact discrete grid FIRST
            E = np.random.randint(-2, 3, size=(num_eq, d_orig))
            import sympy as sp
            null_basis = sp.Matrix(E).nullspace()
            int_basis = []
            for vec in null_basis:
                common_denom = sp.Integer(1)
                for val in vec:
                    common_denom = sp.lcm(common_denom, sp.Rational(val).q)
                int_basis.append(np.array(vec * common_denom, dtype=np.int64).flatten())

            Z_raw = np.column_stack(int_basis)

            # 2. Pick a secret vector guaranteed to be ON the grid
            z_secret = np.random.randint(1, 3, size=Z_raw.shape[1])
            v_secret = Z_raw @ z_secret

            # 3. Generate the 80 tight hyperplanes around it
            B = []
            attempts = 0
            while len(B) < num_ineq and attempts < 10000:
                attempts += 1
                row = np.random.randint(-coeff_range, coeff_range + 1, size=d_orig)
                dot = np.dot(row, v_secret)
                if dot < 0:
                    row, dot = -row, -dot
                if dot > 0:
                    max_dot = max(1, 10 // thinness)
                    if thinness > 1 and dot > max_dot: continue
                    B.append(row)
            B = np.array(B)
            return np.vstack((E, -E, B))
        else:
            raise ValueError("Unknown scenario.")

        if seeding:
            np.random.seed(42 + len(scenario_type))
        v_secret = np.random.randint(1, 4, size=d_orig)

        # 1. Generate Equalities
        E = []
        attempts = 0
        while len(E) < num_eq and attempts < 2000:
            attempts += 1
            row = np.random.randint(-coeff_range, coeff_range + 1, size=d_orig)

            # The Pancake Distortion: Artificially bloat specific dimensions
            if scenario_type == "15D_Pancake":
                distortion_mask = np.random.choice([1, 10, 100], size=d_orig)
                row = row * distortion_mask

            if v_secret[-1] != 0:
                dot_product = np.dot(row[:-1], v_secret[:-1])
                if dot_product % v_secret[-1] == 0:
                    row[-1] = -dot_product // v_secret[-1]
                    if np.any(row != 0):
                        E.append(row)
        E = np.array(E) if len(E) > 0 else np.empty((0, d_orig))

        # 2. Generate Inequalities
        B = []
        attempts = 0
        while len(B) < num_ineq and attempts < 10000:
            attempts += 1
            row = np.random.randint(-coeff_range, coeff_range + 1, size=d_orig)

            dot = np.dot(row, v_secret)
            if dot < 0:
                row, dot = -row, -dot

            if dot > 0:
                # The Needle Distortion: Squeeze the hyperplane tightly against v_secret
                max_dot = max(1, 10 // thinness)
                if thinness > 1 and dot > max_dot:
                    continue
                B.append(row)

        B = np.array(B)

        A_prime_blocks = []
        if len(E) > 0: A_prime_blocks.extend([E, -E])
        if len(B) > 0: A_prime_blocks.append(B)

        return np.vstack(A_prime_blocks)

    def run_gauntlet(self, target_quota=10000, seeding=True):
        scenarios = ["10D_Fat_Baseline", "15D_Needle", "15D_Pancake"]

        for name in scenarios:
            print(f"\n[{name}] Initiating Test Sequence...")
            A_prime = self._generate_gauntlet_matrix(name, seeding)

            try:
                engine = self.EngineClass(A_prime)

                start_time = time.time()
                rays = engine.harvest(target_quota)
                exec_time = time.time() - start_time

                self.results[name] = {
                    "rays": rays,
                    "time": exec_time,
                    "yield": len(rays),
                    "d_flat": engine.d_flat
                }

                print(f"Success: Yielded {len(rays)} rays in {exec_time:.2f}s")
            except Exception as e:
                print(f"FAILED: {str(e)}")
                self.results[name] = {"error": str(e)}

    def render_dashboard(self, scenario_name):
        """
        Generates the 4-pane visual dashboard for a specific scenario run.
        Pane 1: Length Distribution (Histogram)
        Pane 2: Ranked Lengths (Line)
        Pane 3: Angular Uniformity (Nearest Neighbor CDF)
        Pane 4: Radial Uniformity (Volume Q-Q Plot)
        """
        data = self.results.get(scenario_name)
        if not data or "error" in data or len(data["rays"]) == 0:
            print(f"No valid data to plot for {scenario_name}.")
            return

        rays = data["rays"]
        lengths = np.linalg.norm(rays, axis=1)

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        plt.style.use('dark_background')

        # 1. Length Distribution
        axs[0, 0].hist(lengths, bins=50, color='#00FFFF', edgecolor='black')
        axs[0, 0].set_title(f'1. Trajectory Length Distribution for {scenario_name}')

        # 2. Ranked Trajectories
        axs[0, 1].plot(np.sort(lengths), color='#00FF00', linewidth=2)
        axs[0, 1].set_title(f'2. Ranked Trajectories (Shortest to Longest) for {scenario_name}')

        # 3. Angular Uniformity (Nearest Neighbor)
        u_rays = rays / lengths[:, np.newaxis]
        cos_sim = np.clip(u_rays @ u_rays.T, -1.0, 1.0)
        np.fill_diagonal(cos_sim, -1.0)
        min_angles = np.sort(np.degrees(np.arccos(np.max(cos_sim, axis=1))))
        y_cdf = np.arange(1, len(min_angles) + 1) / len(min_angles)
        axs[1, 0].step(min_angles, y_cdf, color='#FF00FF', linewidth=2)
        axs[1, 0].set_title(f'3. Angular Uniformity (NN Angle CDF) for {scenario_name}')

        # 4. Radial Uniformity
        R_max = np.max(lengths)
        emp_quantiles = np.sort((lengths / R_max) ** data["d_flat"])
        theo_quantiles = np.linspace(0, 1, len(emp_quantiles))
        axs[1, 1].step(theo_quantiles, emp_quantiles, color='#5DADE2', label='Empirical')
        axs[1, 1].plot([0,1], [0,1], 'r--', label='Ideal')
        axs[1, 1].set_title(f'4. Radial Volume Uniformity for {scenario_name}')
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.grid(color='grey', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()


def run_diagnostic_dashboard(target_quota=100_000, scenario_name="15D_Realistic_Dense", matrix=None):
    print("="*60)
    print(" INITIATING DIAGNOSTIC HARNESS")
    print("="*60)

    # 1. Initialize the harness with our new unified pipeline
    harness = TestHarness(engine_class=EndToEndSamplingEngine)

    # 2. Generate the specific matrix
    if matrix is not None:
        A_prime = matrix
    else:
        A_prime = harness._generate_gauntlet_matrix(scenario_name)

    # 3. Manually run the test so we can request a large quota
    print(f"\nTargeting {target_quota} rays for {scenario_name}...")

    try:
        engine = harness.EngineClass(A_prime)
        start_time = time.time()
        rays = engine.harvest(target_quota)
        exec_time = time.time() - start_time

        # 4. Store the results in the harness dictionary
        harness.results[scenario_name] = {
            "rays": rays,
            "time": exec_time,
            "yield": len(rays),
            "d_flat": engine.d_flat if hasattr(engine, 'd_flat') else A_prime.shape[1]
            # Note: We use A_prime.shape as a fallback for d_flat for the dashboard
            # if the wrapper doesn't explicitly expose it.
        }

        print(f"\nYAY! Pipeline Complete: Yielded {len(rays)} rays in {exec_time:.2f}s")

        # 5. Render the visual metrics!
        harness.render_dashboard(scenario_name)

    except Exception as e:
        print(f"XXX Harness Execution Failed: {str(e)}")


if __name__ == "__main__":
    # run_diagnostic_dashboard(10_000, "15D_Pancake")
    # run_diagnostic_dashboard(10_000, "15D_Needle")
    # run_diagnostic_dashboard(10_000, "10D_Fat_Baseline")
    np.random.seed(42)
    run_diagnostic_dashboard(1_000, "15D_Realistic_Dense", matrix=np.array([[1, 0, 0], [0, 1, 0], [7, -11, 2], [-7, 11, -2]]))