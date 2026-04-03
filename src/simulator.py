import jax.numpy as jnp
import jax.random as jr
from src.cosmology import generate_cmb_map
from furax.obs.stokes import StokesIQU
from furax.mapmaking.pointing import PointingOperator
from furax.obs import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.math.quaternion import to_gamma_angles
import numpy as np
import torch

def simulate_tod(A_operator, logA, ns, key=None, nside=16, noise_std=5.0):
    if key is None:
        key = jr.PRNGKey(np.random.randint(0, 2**31))
    I_map, Q_map, U_map = generate_cmb_map(logA, ns, nside)
    sky_data = StokesIQU(jnp.array(I_map), jnp.array(Q_map), jnp.array(U_map))
    tod_signal = A_operator(sky_data)
    noise = noise_std * jr.normal(key, tod_signal.shape)
    tod_final = tod_signal + noise
    return tod_final

def wrap_simulator_for_sbi(A_operator, nside=16):
    def simulator(theta):
        logA, ns = theta.cpu().numpy()
        key = jr.PRNGKey(np.random.randint(0, 2**31))
        tod = simulate_tod(A_operator, logA, ns, key, nside)
        return torch.tensor(np.array(tod), dtype=torch.float32)
    return simulator