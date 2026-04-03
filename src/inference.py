import torch
from sbi.inference import SNPE
from sbi import utils as utils

def train_sbi(simulator, prior, theta, x, batch_size=64):
    inference = SNPE(prior=prior)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(training_batch_size=batch_size, show_train_summary=True)
    posterior = inference.build_posterior(density_estimator)
    return posterior