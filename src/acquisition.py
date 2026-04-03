from furax.obs.landscapes import HealpixLandscape
from furax.mapmaking.pointing import PointingOperator
from furax.obs import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.math.quaternion import to_gamma_angles

def build_acquisition_operator(n_detectors, n_samples, bor_quat, det_quat, hwp_angles, nside=16):
    # Create landscape
    landscape = HealpixLandscape(nside=nside, stokes='IQU')
    pointing = PointingOperator.create(landscape, boresight_quaternions=bor_quat, detector_quaternions=det_quat)
    gamma = to_gamma_angles(det_quat)[:, None]
    polarizer = LinearPolarizerOperator.create((n_detectors, n_samples))
    hwp = HWPOperator.create((n_detectors, n_samples), angles=hwp_angles)
    rot = QURotationOperator.create((n_detectors, n_samples), angles=gamma)
    A_operator = polarizer @ rot @ hwp @ pointing
    return A_operator