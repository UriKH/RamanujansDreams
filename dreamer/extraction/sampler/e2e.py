import numpy as np
import scipy.special as spc
from dreamer.extraction.sampler.conditioner import Stage1Conditioner
from dreamer.extraction.sampler.raycaster import Stage2Raycaster
from dreamer.utils.logger import Logger
from typing import Callable


class EndToEndSamplingEngine:
    def __init__(self, A_prime):
        self.A_prime = A_prime
        self.d_orig = A_prime.shape[1]
        self.d_flat = None

    @staticmethod
    def _estimate_cone_fraction(B: np.ndarray, d_flat: int, samples: int = 100_000) -> float:
        """
        Gaussian Measure Dart Throw.
        :param B: Bounds matrix
        :param d_flat: dimension of the flatland space
        :param samples: number of samples - darts to throw.
        """
        if len(B) == 0: return 1.0

        # Pure Gaussian distribution (Spherically symmetric)
        darts = np.random.randn(samples, d_flat)
        darts /= np.linalg.norm(darts, axis=1, keepdims=True)

        valid_count = 0
        for i in range(samples):
            if np.max(B @ darts[i]) <= 1e-9:
                valid_count += 1

        # If the cone is a severe needle, give it a baseline epsilon so our math doesn't divide by zero
        fraction = max(valid_count / samples, 1e-7)
        return fraction

    @staticmethod
    def _calculate_R_max(target_quota: int, fraction: float, d_flat: int) -> float:
        """
        Calculates the theoretical radius needed to hold the quota.
        :param target_quota: target quota of points to sample
        :param fraction: fraction of points to sample
        :param d_flat: dimension of the flatland space
        :return R_max: theoretical radius of the hypersphere containing the target quota samples.
        """
        numerator = target_quota * spc.gamma((d_flat / 2.0) + 1)
        denominator = fraction * (np.pi ** (d_flat / 2.0))
        R_max_d = numerator / denominator
        R_max = R_max_d ** (1.0 / d_flat)
        return R_max

    @staticmethod
    def _verify_uniformity(rays, fraction: float, d_flat: int) -> None:
        """
        Check the generated rays angular uniformity.
        :param rays: The generated rays
        :param fraction: The fraction of space the cone takes
        :param d_flat: The dimension of the sample space
        """
        if len(rays) < 2: return

        # 1. Calculate the dynamic theoretical threshold
        surface_dim = max(1.0, float(d_flat - 1))
        safe_fraction = max(1e-12, fraction)
        theoretical_gap = 180.0 * ((safe_fraction / len(rays)) ** (1.0 / surface_dim))

        # Threshold is 50% of the mathematical ideal
        threshold_degrees = theoretical_gap * 0.5

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

        success = True
        if median_gap < threshold_degrees:
            Logger(f"⚠ WARNING: Severe angular clustering detected. Median NN gap: {median_gap:.2f}°", Logger.Levels.debug).log()
            success = False
        else:
            Logger(f"Uniformity Check Passed: Healthy angular separation. Median NN gap: {median_gap:.2f}°", Logger.Levels.debug).log()

        if mean_gap < threshold_degrees:
            Logger(f"⚠ WARNING: Severe angular clustering detected. Mean NN gap: {mean_gap:.2f}°", Logger.Levels.debug).log()
            success = False
        else:
            Logger(f"Uniformity Check Passed: Healthy angular separation. Mean NN gap: {mean_gap:.2f}°", Logger.Levels.debug).log()

        if not success:
            Logger(f"Could not preform uniform sampling as expected... (if this repeats many times please report)", Logger.Levels.warning).log()

    def harvest(self, target_func: Callable[[int], int] | int, guidance_method: str = 'mcmc') -> np.ndarray:
        """
        Harvest samples
        :param target_func: Target function to compute total expected quota
        :param guidance_method: Ray sampling guidance method - MCMC or MHS
        :return: The samples
        """
        Logger("[Pipeline] Initializing Stage 1: Conditioning...", Logger.Levels.debug).log()
        conditioner = Stage1Conditioner(self.A_prime, max_beta=10, defect_tolerance=5.0)

        try:
            Z_reduced, B_reduced, _ = conditioner.process()
        except ValueError as e:
            raise Exception(f"[Pipeline] Stage 1 Failed: {e}")
            return np.array([])

        self.d_flat = Z_reduced.shape[1]

        fraction = self._estimate_cone_fraction(B_reduced, self.d_flat)
        Logger(f"[Pipeline] Cone Volume Estimate: {fraction*100:.6f}% of total sphere.", Logger.Levels.debug).log()

        if type(target_func) == int:
            target_rays = target_func
        else:
            amount_safety = 1.05
            target_rays = max(int(target_func(self.d_flat) * fraction * amount_safety), 5)
        R_max = self._calculate_R_max(target_rays, fraction, self.d_flat)
        Logger(f"[Pipeline] Mathematical R_max needed for {target_rays} rays: {R_max:.2f}", Logger.Levels.debug).log()
        Logger("[Pipeline] Initializing Stage 2: Universal Raycaster...", Logger.Levels.debug).log()
        sampler = Stage2Raycaster(Z_reduced, B_reduced, self.d_orig, guidance_method)

        # Oversample by 3x
        guide_rays_to_shoot = int(target_rays * 3)
        current_R_max = R_max * 1.05
        final_rays = np.empty((0, self.d_orig))

        if self.d_flat >= 4:
            # Massive outer shell. strictly enforce Fair Slice (1 point per ray)
            dynamic_max_per_ray = 1
        else:
            # Microscopic outer shell. Must penetrate deep to fill quota.
            dynamic_max_per_ray = max(1, int(1.5 * (target_rays ** (1.0 / self.d_flat))))
            Logger(f"[Pipeline] Low-D Space Detected. Allowing depth penetration: max_per_ray = {dynamic_max_per_ray}", Logger.Levels.debug).log()

        while len(final_rays) < target_rays:
            Logger(f"Sweeping lattice up to R_max = {current_R_max:.2f}...", Logger.Levels.debug).log()

            # Enforce max_per_ray=1 for the "Fair Slice"
            raw_rays = sampler.harvest(
                target_rays=guide_rays_to_shoot,
                R_max=current_R_max,
                max_per_ray=dynamic_max_per_ray
            )

            if len(raw_rays) >= target_rays:
                Logger(f"Quota exceeded ({len(raw_rays)}). Engaging Expanding Ball (Length Sort)...", Logger.Levels.debug).log()
                lengths = np.linalg.norm(raw_rays, axis=1)
                sorted_indices = np.argsort(lengths)
                best_rays = raw_rays[sorted_indices][:target_rays]
                np.random.shuffle(best_rays)
                final_rays = best_rays
                break
            else:
                if len(raw_rays) == 0:
                    momentum_multiplier = 2.0
                else:
                    # Dimensional scaling law: V_new / V_old = R_multiplier ^ d_flat
                    ratio_needed = target_rays / len(raw_rays)
                    momentum_multiplier = ratio_needed ** (1.0 / self.d_flat)

                # Cap the multiplier between 1.10 (minimum safety step) and 3.0 (max jump)
                multiplier = np.clip(momentum_multiplier, 1.10, 3.0)
                Logger(
                    f"Discretization Gap hit: Yielded {len(raw_rays)} bounded rays. Target: {target_rays}.",
                    Logger.Levels.debug
                ).log()
                Logger(f"\n   -> Momentum Expansion: Multiplying R_max by {multiplier:.3f}", Logger.Levels.debug).log()
                current_R_max *= multiplier

        self._verify_uniformity(final_rays, fraction, self.d_flat)
        return final_rays