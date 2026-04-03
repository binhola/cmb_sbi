import torch
from sbi import utils as utils
from src.inference import train_sbi
from src.simulator import wrap_simulator_for_sbi
import numpy as np

# Load TOD
tod = np.load("data/tod/tod.npy")
n_detectors, n_samples = tod.shape

# Define prior
low = torch.tensor([2.0, 0.9])
high = torch.tensor([4.0, 1.0])
prior = utils.BoxUniform(low=low, high=high)

# Wrap simulator
# (replace A_operator with pre-built or dummy for now)
A_operator = None  # TODO: load real operator
simulator = wrap_simulator_for_sbi(A_operator)

# Generate small training set
num_simulations = 100
theta = prior.sample((num_simulations,))
x = [simulator(theta[i]) for i in range(num_simulations)]
x = torch.stack(x)

posterior = train_sbi(simulator, prior, theta, x, batch_size=16)
torch.save(posterior, "data/simulations/posterior.pt")
print("Posterior trained and saved.")