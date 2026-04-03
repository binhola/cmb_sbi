import yaml
from src.cosmology import generate_cmb_map
import numpy as np
from pathlib import Path

# Load config
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

nside = cfg['nside']
maps_dir = Path("data/maps")
maps_dir.mkdir(parents=True, exist_ok=True)

# Example: generate one map
I_map, Q_map, U_map = generate_cmb_map(logA=3.04, ns=0.965, nside=nside)

np.save(maps_dir / "I_map.npy", I_map)
np.save(maps_dir / "Q_map.npy", Q_map)
np.save(maps_dir / "U_map.npy", U_map)

print("CMB maps generated and saved.")