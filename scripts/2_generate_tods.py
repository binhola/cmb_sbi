import yaml
import jax.random as jr
import numpy as np
from pathlib import Path
from furax.interfaces.toast.observation import ToastObservation
from src.acquisition import build_acquisition_operator
from src.simulator import simulate_tod
import jax.numpy as jnp

# Load config
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

n_detectors = cfg['n_detectors']
nside = cfg['nside']
noise_std = cfg['noise_std']

# Path relative to the script location
filename = Path(__file__).resolve().parent.parent / "data" / "toast" / "test_obs.h5"

if not filename.exists():
    raise FileNotFoundError(f"{filename} does not exist")

obs = ToastObservation.from_file(filename)

# Dummy quaternions for example
bor_quat = obs.get_boresight_quaternions() # shape: (n_samples, 4)
det_quat = obs.get_detector_quaternions() # shape: (n_detector, 4)
n_samples = bor_quat.shape[0]

hwp_angles = 2 * np.pi * cfg['hwp_freq'] * np.arange(n_samples) / n_samples

A_operator = build_acquisition_operator(n_detectors, n_samples, bor_quat, det_quat, hwp_angles, nside)

tod_dir = Path("data/tod")
tod_dir.mkdir(parents=True, exist_ok=True)

key = jr.PRNGKey(0)
tod = simulate_tod(A_operator, logA=3.04, ns=0.965, key=key, nside=nside, noise_std=noise_std)

np.save(tod_dir / "tod.npy", np.array(tod))
print("TOD generated and saved.")