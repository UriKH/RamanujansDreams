import numpy as np
import scipy.special as spc
from dreamer.extraction.sampler.conditioner import Stage1_Conditioner
from dreamer.extraction.sampler.raycaster import Stage2_Raycaster
from typing import Callable


class EndToEndSamplingEngine:
    def __init__(self, A_prime):
        self.A_prime = A_prime
        self.d_orig = A_prime.shape[1]
        self.d_flat = None

    @staticmethod
    def _estimate_cone_fraction(B, d_flat, samples=100000):
        """Gaussian Measure Dart Throw."""
        if len(B) == 0: return 1.0

        # Pure Gaussian distribution (Spherically symmetric)
        darts = np.random.randn(samples, d_flat)
        darts /= np.linalg.norm(darts, axis=1, keepdims=True)

        valid_count = 0
        for i in range(samples):
            if np.max(B @ darts[i]) <= 1e-9:
                valid_count += 1

        # If the cone is a severe needle, give it a baseline epsilon
        # so our math doesn't divide by zero
        fraction = max(valid_count / samples, 1e-7)
        return fraction

    @staticmethod
    def _calculate_R_max(target_quota, fraction, d_flat):
        """Calculates the theoretical radius needed to hold the quota."""
        # V_d(R) = (pi^(d/2) / Gamma(d/2 + 1)) * R^d
        numerator = target_quota * spc.gamma((d_flat / 2.0) + 1)
        denominator = fraction * (np.pi ** (d_flat / 2.0))

        R_max_d = numerator / denominator
        R_max = R_max_d ** (1.0 / d_flat)
        return R_max

    @staticmethod
    def _verify_uniformity(rays, threshold_degrees=1.0):
        if len(rays) < 2: return

        # Subsample to keep the matrix math lightning fast (max 2000 points)
        sample_size = min(2000, len(rays))
        sample = rays[np.random.choice(len(rays), sample_size, replace=False)]

        # Normalize and compute cosine similarity matrix
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        normalized = sample / np.clip(norms, 1e-9, None)
        cos_sim = np.clip(normalized @ normalized.T, -1.0, 1.0)

        # Ignore self-similarity (diagonal)
        np.fill_diagonal(cos_sim, -1.0)
        max_sim = np.max(cos_sim, axis=1)

        # Convert to degrees
        min_angles = np.arccos(max_sim) * (180.0 / np.pi)
        median_gap = np.median(min_angles)
        mean_gap = np.mean(min_angles)

        if median_gap < threshold_degrees:
            print(f"⚠ WARNING: Severe angular clustering detected. Median NN gap: {median_gap:.2f}°")
        # else:
        #     print(f":) Uniformity Check Passed: Healthy angular separation. Median NN gap: {median_gap:.2f}°")

        if mean_gap < threshold_degrees:
            print(f"⚠ WARNING: Severe angular clustering detected. Mean NN gap: {mean_gap:.2f}°")
        # else:
        #     print(f":) Uniformity Check Passed: Healthy angular separation. Mean NN gap: {mean_gap:.2f}°")

    def harvest(self, target_func: Callable[[int], int] | int, guidance_method: str = 'mcmc'):
        # print("\n[Pipeline] Initializing Stage 1: Conditioning...")
        conditioner = Stage1_Conditioner(self.A_prime, max_beta=10, defect_tolerance=5.0)

        try:
            Z_reduced, B_reduced, _ = conditioner.process()
        except ValueError as e:
            raise Exception(f"!X! [Pipeline] Stage 1 Failed: {e}")
            return np.array([])

        self.d_flat = Z_reduced.shape[1]

        fraction = self._estimate_cone_fraction(B_reduced, self.d_flat)
        # print(f"[Pipeline] Cone Volume Estimate: {fraction*100:.6f}% of total sphere.")

        if type(target_func) == int:
            target_rays = target_func
        else:
            amount_safety = 1.05
            target_rays = max(int(target_func(self.d_flat) * fraction * amount_safety), 5)
        R_max = self._calculate_R_max(target_rays, fraction, self.d_flat)
        # print(f"[Pipeline] Mathematical R_max needed for {target_rays} rays: {R_max:.2f}")

        # print("\n[Pipeline] Initializing Stage 2: Universal Raycaster...")
        sampler = Stage2_Raycaster(Z_reduced, B_reduced, self.d_orig, guidance_method)

        # Oversample by 3x so we have a massive pool to select the absolute shortest from
        guide_rays_to_shoot = int(target_rays * 3)
        current_R_max = R_max * 1.05
        final_rays = np.empty((0, self.d_orig))

        if self.d_flat >= 4:
            # High dimensions: Massive outer shell. strictly enforce Fair Slice (1 point per ray)
            dynamic_max_per_ray = 1
        else:
            # Low dimensions: Microscopic outer shell. Must penetrate deep to fill quota.
            dynamic_max_per_ray = max(1, int(1.5 * (target_rays ** (1.0 / self.d_flat))))
            # print(f"[Pipeline] Low-D Space Detected. Allowing depth penetration: max_per_ray = {dynamic_max_per_ray}")

        while len(final_rays) < target_rays:
            # print(f">>> Sweeping lattice up to R_max = {current_R_max:.2f}...")

            # Enforce max_per_ray=1 for the "Fair Slice"
            raw_rays = sampler.harvest(
                target_rays=guide_rays_to_shoot,
                R_max=current_R_max,
                max_per_ray=dynamic_max_per_ray
            )

            if len(raw_rays) >= target_rays:
                # print(f":) Quota exceeded ({len(raw_rays)}). Engaging Expanding Ball (Length Sort)...")
                # THE "EXPANDING BALL" LOGIC
                lengths = np.linalg.norm(raw_rays, axis=1)
                sorted_indices = np.argsort(lengths)

                # Take the absolute shortest N rays
                best_rays = raw_rays[sorted_indices][:target_rays]

                # Shuffle the final array to preserve a random angular feed for your downstream process
                np.random.shuffle(best_rays)
                final_rays = best_rays
                break
            else:
                # print(f"Discretization Gap hit: Yielded {len(raw_rays)} bounded rays. Target: {target_rays}.")
                # print("   Expanding R_max boundary by 15% and re-casting...")
                # current_R_max *= 1.15
                if len(raw_rays) == 0:
                    momentum_multiplier = 2.0 # Extreme jump if totally empty
                else:
                    # Dimensional scaling law: V_new / V_old = R_multiplier ^ d_flat
                    ratio_needed = target_rays / len(raw_rays)
                    momentum_multiplier = ratio_needed ** (1.0 / self.d_flat)

                # Cap the multiplier between 1.10 (minimum safety step) and 3.0 (max jump)
                multiplier = np.clip(momentum_multiplier, 1.10, 3.0)

                # print(f"   -> Momentum Expansion: Multiplying R_max by {multiplier:.3f}")
                current_R_max *= multiplier

        self._verify_uniformity(final_rays)
        return final_rays