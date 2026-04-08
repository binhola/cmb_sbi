import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
import numpy as np
import torch

import camb
import healpy as hp

from furax.obs.stokes import StokesIQU
from furax.obs.landscapes import HealpixLandscape
from furax.obs.pointing import PointingOperator
from furax.obs import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.math.quaternion import to_gamma_angles
from furax.interfaces.toast.observation import ToastObservation

from sbi.inference import SNPE
from sbi import utils as utils

# -------------------------
# Parameters
# -------------------------
nside = 16
num_simulations = 500
noise_std = 5.0

# -------------------------
# Load observation & operator
# -------------------------
filename = "data/toast/test_obs.h5"
obs = ToastObservation.from_file(filename)

tod = obs.get_tods()
bor_quat = obs.get_boresight_quaternions()
det_quat = obs.get_detector_quaternions()
hwp_angles = obs.get_hwp_angles()

landscape = HealpixLandscape(nside=nside, stokes='IQU')
pointing = PointingOperator.create(landscape, boresight_quaternions=bor_quat, detector_quaternions=det_quat)
gamma = to_gamma_angles(det_quat)[:, None]

polarizer = LinearPolarizerOperator.create(tod.shape)
hwp = HWPOperator.create(tod.shape, angles=hwp_angles)
rot = QURotationOperator.create(tod.shape, angles=gamma)

A = polarizer @ rot @ hwp @ pointing

# -------------------------
# 1. CMB map generator (Python/NumPy)
# -------------------------
def generate_cmb_map(logA, ns, nside=nside):
    As = np.exp(logA) / 1e10
    pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06,
                           omk=0, tau=0.06, As=As, ns=ns, halofit_version='mead', lmax=3*nside)
    pars.Lensing = False
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    cls = powers['total']
    lmax = 3*nside - 1
    Cl_TT, Cl_EE, Cl_BB, Cl_TE = cls[:lmax+1,0], cls[:lmax+1,1], cls[:lmax+1,2], cls[:lmax+1,3]
    I_map, Q_map, U_map = hp.synfast((Cl_TT, Cl_EE, Cl_BB, Cl_TE), nside=nside, lmax=lmax, pol=True)
    return I_map, Q_map, U_map

# -------------------------
# 2. JIT TOD operator (JAX)
# -------------------------
@jit
def tod_operator(I_jax, Q_jax, U_jax, key):
    sky = StokesIQU(I_jax, Q_jax, U_jax)
    tod_signal = A(sky)
    noise = noise_std * jr.normal(key, tod_signal.shape)
    return tod_signal + noise

# -------------------------
# 3. Simulator wrapper
# -------------------------
def simulator(theta):
    logA, ns = theta.cpu().numpy()
    I_map, Q_map, U_map = generate_cmb_map(logA, ns, nside=nside)
    I_jax, Q_jax, U_jax = jnp.array(I_map), jnp.array(Q_map), jnp.array(U_map)
    key = jr.PRNGKey(np.random.randint(0, 2**31))
    tod_jax = tod_operator(I_jax, Q_jax, U_jax, key)
    return torch.tensor(np.array(tod_jax), dtype=torch.float32)

# -------------------------
# 4. Vectorized simulator (optional, batch)
# -------------------------
def batch_simulator(thetas):
    keys = jr.split(jr.PRNGKey(np.random.randint(0,2**31)), len(thetas))
    tods = []
    for theta, key in zip(thetas, keys):
        I_map, Q_map, U_map = generate_cmb_map(theta[0], theta[1], nside=nside)
        I_jax, Q_jax, U_jax = jnp.array(I_map), jnp.array(Q_map), jnp.array(U_map)
        tods.append(tod_operator(I_jax, Q_jax, U_jax, key))
    return torch.tensor(np.array(tods), dtype=torch.float32)

# -------------------------
# 5. Prior and SNPE
# -------------------------
low = torch.tensor([2.0, 0.9])
high = torch.tensor([4.0, 1.0])
prior = utils.BoxUniform(low=low, high=high)

theta = prior.sample((num_simulations,))
print("Generating simulations...")
x = batch_simulator(theta)

print("Training SNPE...")
inference = SNPE(prior=prior)
inference = inference.append_simulations(theta, x)
density_estimator = inference.train(training_batch_size=64, show_train_summary=True)
posterior = inference.build_posterior(density_estimator)

# -------------------------
# 6. Test on a mock TOD
# -------------------------
true_logA, true_ns = 3.04, 0.965
mock_tod = simulator(torch.tensor([true_logA, true_ns]))
samples = posterior.sample((5000,), x=mock_tod)

# -------------------------
# 7. Plot posterior
# -------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(samples[:,0].numpy(), bins=30, density=True, alpha=0.7)
axes[0].axvline(true_logA, color='r', linestyle='--', label='True')
axes[0].set_xlabel(r'$\ln(10^{10} A_s)$')
axes[0].set_ylabel('Posterior density')
axes[0].legend()

axes[1].hist(samples[:,1].numpy(), bins=30, density=True, alpha=0.7)
axes[1].axvline(true_ns, color='r', linestyle='--', label='True')
axes[1].set_xlabel(r'$n_s$')
axes[1].legend()

plt.suptitle('SNPE Posterior Inference')
plt.tight_layout()
plt.savefig("results/test_1.png")