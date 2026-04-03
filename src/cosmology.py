import camb
import healpy as hp
import numpy as np

def generate_cmb_map(logA, ns, nside=16):
    As = np.exp(logA) / 1e10
    pars = camb.set_params(
        H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,
        As=As, ns=ns, halofit_version='mead', lmax=3*nside
    )
    pars.Lensing = False
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    cls = powers['total']
    lmax = 3 * nside - 1
    Cl_TT = cls[:lmax+1, 0]
    Cl_EE = cls[:lmax+1, 1]
    Cl_BB = cls[:lmax+1, 2]
    Cl_TE = cls[:lmax+1, 3]
    I_map, Q_map, U_map = hp.synfast((Cl_TT, Cl_EE, Cl_BB, Cl_TE), nside=nside, lmax=lmax, pol=True)
    return I_map, Q_map, U_map