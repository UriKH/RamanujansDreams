import scipy.optimize as opt
from numba import njit, prange
import time
import numpy as np


@njit(parallel=True)
def _raycast_kernel_parallel(d_orig, d_flat, Z_int, B, continuous_rays, R_max, t_step=0.1, max_per_ray=5):
    num_rays = len(continuous_rays)
    m = len(B)

    # Pre-allocate the maximum possible size.
    harvest_buffer = np.zeros((num_rays * max_per_ray, d_orig), dtype=np.int64)
    # Track how many points EACH ray found independently
    ray_counts = np.zeros(num_rays, dtype=np.int32)

    # Launch parallel threads!
    for r_idx in prange(num_rays):
        v_dir = continuous_rays[r_idx]
        base_idx = r_idx * max_per_ray # The exact memory lane for this specific ray
        hits = 0

        t = 1.0
        while t <= R_max:
            p = t * v_dir

            z = np.zeros(d_flat, dtype=np.int64)
            is_origin = True
            for j in range(d_flat):
                val = int(np.round(p[j]))
                z[j] = val
                if val != 0: is_origin = False

            if is_origin:
                t += t_step
                continue

            valid = True
            for i in range(m):
                val = 0.0
                for j in range(d_flat): val += B[i, j] * z[j]
                if val > 1e-9:
                    valid = False
                    break

            if valid:
                v_real = np.zeros(d_orig, dtype=np.int64)
                for i in range(d_orig):
                    for j in range(d_flat): v_real[i] += Z_int[i, j] * z[j]

                a = abs(v_real[0])
                for i in range(1, d_orig):
                    b = abs(v_real[i])
                    while b: a, b = b, a % b

                if a == 1:
                    is_new = True
                    if hits > 0:
                        prev = harvest_buffer[base_idx + hits - 1]
                        match = True
                        for dim in range(d_orig):
                            if prev[dim] != v_real[dim]: match = False; break
                        if match: is_new = False

                    if is_new:
                        harvest_buffer[base_idx + hits] = v_real
                        hits += 1

                        if hits >= max_per_ray: break

            t += t_step

        ray_counts[r_idx] = hits # Lock in this thread's count

    return harvest_buffer, ray_counts

# @njit(parallel=True)
# def _generate_guide_rays_kernel(d_flat, B, start_pos, target_rays, mix_steps=200):
#     rays = np.zeros((target_rays, d_flat), dtype=np.float64)
#     m = len(B)
#
#     for i in prange(target_rays):
#         pos = start_pos.copy()
#
#         for _ in range(mix_steps):
#             # 1. Random Direction
#             v = np.random.randn(d_flat)
#             norm_v = 0.0
#             for j in range(d_flat): norm_v += v[j]*v[j]
#             norm_v = np.sqrt(norm_v)
#             for j in range(d_flat): v[j] /= norm_v
#
#             # 2. Raycast against 80 hyperplanes
#             t_min = -1e12
#             t_max = 1e12
#
#             for j in range(m):
#                 dot_v = 0.0
#                 dot_p = 0.0
#                 for k in range(d_flat):
#                     dot_v += B[j, k] * v[k]
#                     dot_p += B[j, k] * pos[k]
#
#                 if dot_v > 1e-9:
#                     t = -dot_p / dot_v
#                     if t < t_max: t_max = t
#                 elif dot_v < -1e-9:
#                     t = -dot_p / dot_v
#                     if t > t_min: t_min = t
#
#             # 3. Move continuous walker
#             if t_max > t_min:
#                 t_step = np.random.uniform(t_min, t_max)
#                 for k in range(d_flat):
#                     pos[k] += t_step * v[k]
#             else:
#                 for k in range(d_flat):
#                     pos[k] *= 0.99
#
#         # 4. Finalize the ray (normalize position to get pure direction)
#         norm_pos = 0.0
#         for k in range(d_flat): norm_pos += pos[k]*pos[k]
#         norm_pos = np.sqrt(norm_pos)
#
#         if norm_pos > 0:
#             for k in range(d_flat):
#                 rays[i, k] = pos[k] / norm_pos
#         else:
#             for k in range(d_flat):
#                 rays[i, k] = v[k]
#
#     return rays

@njit(parallel=True)
def _generate_guide_rays_mcmc_kernel(d_flat, B, start_pos, target_rays, mix_steps=200):
    rays = np.zeros((target_rays, d_flat), dtype=np.float64)
    m = len(B)
    num_chains = 16
    rays_per_chain = (target_rays // num_chains) + 1

    for chain_idx in prange(num_chains):
        pos = start_pos.copy()
        start_norm = 0.0
        for k in range(d_flat): start_norm += pos[k]*pos[k]
        if start_norm > 0:
            start_norm = np.sqrt(start_norm)
            for k in range(d_flat): pos[k] /= start_norm

        for _ in range(1000): # Burn-in
            v = np.random.randn(d_flat)
            norm_v = 0.0
            for j in range(d_flat): norm_v += v[j]*v[j]
            norm_v = np.sqrt(norm_v)
            for j in range(d_flat): v[j] /= norm_v

            # Standard Hit-and-Run on Hyperplanes
            t_min, t_max = -1e9, 1e9
            for j in range(m):
                dot_v = 0.0; dot_p = 0.0
                for k in range(d_flat):
                    dot_v += B[j, k] * v[k]
                    dot_p += B[j, k] * pos[k]
                if dot_v > 1e-9:
                    t = -dot_p / dot_v
                    if t < t_max: t_max = t
                elif dot_v < -1e-9:
                    t = -dot_p / dot_v
                    if t > t_min: t_min = t

            # Cap maximum jump to prevent pure infinity drift in fully open cones
            if t_max > 100.0: t_max = 100.0
            if t_min < -100.0: t_min = -100.0

            if t_max > t_min:
                t_step = np.random.uniform(t_min, t_max)
                for k in range(d_flat): pos[k] += t_step * v[k]
            else:
                for k in range(d_flat): pos[k] *= 0.99

            # THE FIX: Simply project back to Unit Sphere after jumping
            norm_pos = 0.0
            for k in range(d_flat): norm_pos += pos[k]*pos[k]
            if norm_pos > 0:
                norm_pos = np.sqrt(norm_pos)
                for k in range(d_flat): pos[k] /= norm_pos

        # Record Breadcrumbs
        for i in range(rays_per_chain):
            global_idx = chain_idx * rays_per_chain + i
            if global_idx >= target_rays: break

            for _ in range(mix_steps):
                v = np.random.randn(d_flat)
                norm_v = 0.0
                for j in range(d_flat): norm_v += v[j]*v[j]
                norm_v = np.sqrt(norm_v)
                for j in range(d_flat): v[j] /= norm_v

                t_min, t_max = -1e9, 1e9
                for j in range(m):
                    dot_v = 0.0; dot_p = 0.0
                    for k in range(d_flat):
                        dot_v += B[j, k] * v[k]
                        dot_p += B[j, k] * pos[k]
                    if dot_v > 1e-9:
                        t = -dot_p / dot_v
                        if t < t_max: t_max = t
                    elif dot_v < -1e-9:
                        t = -dot_p / dot_v
                        if t > t_min: t_min = t

                if t_max > 100.0: t_max = 100.0
                if t_min < -100.0: t_min = -100.0

                if t_max > t_min:
                    t_step = np.random.uniform(t_min, t_max)
                    for k in range(d_flat): pos[k] += t_step * v[k]
                else:
                    for k in range(d_flat): pos[k] *= 0.99

                # Project back to sphere
                norm_pos = 0.0
                for k in range(d_flat): norm_pos += pos[k]*pos[k]
                if norm_pos > 0:
                    norm_pos = np.sqrt(norm_pos)
                    for k in range(d_flat): pos[k] /= norm_pos

            for k in range(d_flat): rays[global_idx, k] = pos[k]

    return rays


@njit(parallel=True)
def _generate_guide_rays_mhs_kernel(d_flat, B, start_pos, target_rays, mix_steps=200):
    """
    Sampler based on Metropolis-Hastings-Surface walker
    :param d_flat:
    :param B:
    :param start_pos:
    :param target_rays:
    :param mix_steps:
    :return:
    """
    rays = np.zeros((target_rays, d_flat), dtype=np.float64)
    m = len(B)
    num_chains = 16
    rays_per_chain = (target_rays // num_chains) + 1

    for chain_idx in prange(num_chains):
        # 1. Initialize walker exactly on the sphere surface
        pos = start_pos.copy()
        new_pos = np.zeros(d_flat, dtype=np.float64)

        start_norm = 0.0
        for k in range(d_flat): start_norm += pos[k]*pos[k]
        if start_norm > 0:
            start_norm = np.sqrt(start_norm)
            for k in range(d_flat): pos[k] /= start_norm

        # Total steps = Burn-in + (Number of rays * Mix steps)
        total_steps = 1000 + rays_per_chain * mix_steps
        ray_count = 0

        for step_idx in range(total_steps):
            # 2. Adaptive Multi-Scale Jump (Crucial for Dimension Agnosticism)
            rand_val = np.random.rand()
            if rand_val < 0.2: sigma = 0.5      # Big jump (Sweeps wide 2D cones)
            elif rand_val < 0.5: sigma = 0.1    # Medium jump
            elif rand_val < 0.8: sigma = 0.01   # Small jump (Navigates 15D Needles)
            else: sigma = 0.001                 # Micro jump (Survives 15D Pancakes)

            # 3. Propose a new step on the surface of the sphere
            norm_new = 0.0
            for k in range(d_flat):
                new_pos[k] = pos[k] + sigma * np.random.randn()
                norm_new += new_pos[k] * new_pos[k]

            norm_new = np.sqrt(norm_new)
            for k in range(d_flat): new_pos[k] /= norm_new

            # 4. Check boundaries (Hyperplanes)
            valid = True
            for j in range(m):
                dot_val = 0.0
                for k in range(d_flat): dot_val += B[j, k] * new_pos[k]
                if dot_val > 1e-9:
                    valid = False
                    break

            # 5. Metropolis-Hastings Rule: Accept if valid, stay in place if invalid
            if valid:
                for k in range(d_flat): pos[k] = new_pos[k]

            # 6. Record Breadcrumb
            if step_idx >= 1000 and (step_idx - 1000) % mix_steps == 0:
                global_idx = chain_idx * rays_per_chain + ray_count
                if global_idx < target_rays:
                    for k in range(d_flat): rays[global_idx, k] = pos[k]
                    ray_count += 1

    return rays


class Stage2_Raycaster:
    def __init__(self, Z_reduced, B_reduced, d_orig, guidance_method='mcmc'):
        self.Z = np.array(Z_reduced, dtype=np.int64)
        self.B = np.array(B_reduced, dtype=np.float64)
        self.d_orig = d_orig
        self.d_flat = self.Z.shape[1]

        if guidance_method == 'mcmc':
            self.guidance_method = _generate_guide_rays_mcmc_kernel
        elif guidance_method == 'mhs':
            self.guidance_method = _generate_guide_rays_mhs_kernel
        else:
            raise ValueError('Unknown guidance method')

    def _get_chebyshev_center(self):
        m, d = self.B.shape
        c = np.zeros(d + 1)
        c[-1] = -1.0

        A_ub = np.zeros((m, d + 1))
        A_ub[:, :-1] = self.B
        A_ub[:, -1] = np.linalg.norm(self.B, axis=1)

        b_ub = np.full(m, -1e-5)
        bounds = [(-1.0, 1.0) for _ in range(d)] + [(0, None)]

        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.x[-1] > 1e-9:
            return res.x[:-1]
        return None

    def _generate_continuous_guide_rays(self, target_rays, mix_steps=200):
        """Ultra-fast parallel C-kernel generation of continuous guide rays."""
        start_pos = self._get_chebyshev_center()
        if start_pos is None: return None

        # Execute the Numba kernel
        rays = self.guidance_method(
            self.d_flat, self.B, start_pos, target_rays, mix_steps
        )
        return rays

    def harvest(self, target_rays, R_max, max_per_ray=1):
        if self.d_flat == 0: return np.array([])

        # print(f"Raycaster: Generating {target_rays} Continuous Guide Rays...")
        guide_rays = self._generate_continuous_guide_rays(target_rays)
        if guide_rays is None:
            # print("XXX Closed Cone.")
            return np.array([])

        # print(f"Raycaster: Sweeping lattice along Guide Rays...")
        start_t = time.time()

        # Pass max_per_ray into the Numba kernel
        raw_buffer, counts = _raycast_kernel_parallel(
            self.d_orig, self.d_flat, self.Z, self.B, guide_rays, R_max, max_per_ray=max_per_ray
        )

        # FIXED: Use max_per_ray dynamically instead of the hardcoded 5
        valid_blocks = [raw_buffer[i * max_per_ray : i * max_per_ray + counts[i]]
                        for i in range(len(counts)) if counts[i] > 0]

        if not valid_blocks:
            return np.empty((0, self.d_orig))

        merged = np.vstack(valid_blocks)
        unique_rays = np.unique(merged, axis=0)

        # print(f"Raycaster Yielded {len(unique_rays)} unique, shortest trajectories in {time.time()-start_t:.3f}s")
        return unique_rays