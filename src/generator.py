# src/generator.py

import os
import torch
import numpy as np
import healpy as hp
import camb
from pathlib import Path
from tqdm import tqdm

from sbi.utils import BoxUniform
from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import StokesI
from furax.obs.pointing import PointingOperator
from furax.interfaces.toast.observation import ToastObservation


# -----------------------------
# CMB power spectrum
# -----------------------------
def get_cl_tt_for_As(As, lmax=3000):
    pars = camb.set_params(
        H0=67.5, ombh2=0.022, omch2=0.122,
        mnu=0.06, omk=0, tau=0.06, ns=0.965, lmax=lmax
    )
    pars.InitPower.set_params(As=As)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

    return powers['total'][:, 0]


# -----------------------------
# Load pointing operator
# -----------------------------
def load_pointing(nside):
    project_root = Path("/global/u2/b/binguyen/cmb_sbi")  # your full project folder
    filename = project_root / "data" / "toast" / "test_obs.h5"

    if not filename.exists():
        raise FileNotFoundError(f"{filename} not found")

    obs = ToastObservation.from_file(filename)

    landscape = HealpixLandscape(nside=nside, stokes='I')

    pointing = PointingOperator.create(
        landscape,
        boresight_quaternions=obs.get_boresight_quaternions(),
        detector_quaternions=obs.get_detector_quaternions()
    )

    return pointing


# -----------------------------
# Simulator
# -----------------------------
def simulator(As, cl_ref, As_ref, pointing, nside, sigma_noise=None, return_map=False):
    # Scale Cl
    cl_tt = (As / As_ref) * cl_ref

    # Map
    I_map = hp.synfast(cl_tt, nside=nside)
    I_map = StokesI(I_map)

    # TOD
    tod = pointing(I_map)
    signal = tod.i

    # Optional noise
    if sigma_noise is not None:
        noise = np.random.normal(0, sigma_noise, size=signal.shape)
        signal = signal + noise
        
    if return_map:
        return torch.tensor(signal, dtype=torch.float32), I_map
        
    return torch.tensor(signal, dtype=torch.float32)


# -----------------------------
# Main generator
# -----------------------------
def generate_dataset(
    num_simulations=1000,
    nside=512,
    save_dir="data/simulations",
    sigma_noise=None,
    seed=0
):
    project_root = Path("/global/u2/b/binguyen/cmb_sbi")
    save_dir = project_root / save_dir
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    # Load pointing
    print("Loading pointing...")
    pointing = load_pointing(nside)

    # Prior
    prior = BoxUniform(
        low=torch.tensor([5e-10]),
        high=torch.tensor([5e-9])
    )

    theta = prior.sample((num_simulations,))

    # Reference Cl
    As_ref = 2.75e-9
    print(f"Computing reference Cl for As_ref = {As_ref:.2e}")
    cl_ref = get_cl_tt_for_As(As_ref)

    # Simulations
    x_list = []

    print("Generating simulations...")
    for i in tqdm(range(num_simulations)):
        As = theta[i].item()

        x = simulator(
            As,
            cl_ref=cl_ref,
            As_ref=As_ref,
            pointing=pointing,
            nside=nside,
            sigma_noise=sigma_noise
        )

        x_list.append(x)

    x = torch.stack(x_list)

    # Save
    torch.save(theta, os.path.join(save_dir, "theta.pt"))
    torch.save(x, os.path.join(save_dir, "x.pt"))

    print(f"Saved dataset to {save_dir}")

    return theta, x

def generate_observation(
    As_true,
    nside,
    sigma_noise=None,
    return_map=False
):
    # Load pointing once
    pointing = load_pointing(nside=nside)

    # Compute reference C_l once
    As_ref = 2.75e-9
    cl_ref = get_cl_tt_for_As(As_ref)
    
    x_obs = simulator(
        As_true,
        cl_ref=cl_ref,
        As_ref=As_ref,
        pointing=pointing,
        nside=nside,
        sigma_noise=sigma_noise,
        return_map=return_map
    )

    return x_obs