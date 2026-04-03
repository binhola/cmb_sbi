import torch
import matplotlib.pyplot as plt
import numpy as np

posterior = torch.load("data/simulations/posterior.pt")

true_logA = 3.04
true_ns = 0.965

# TODO: load corresponding TOD
mock_tod = None  # Replace with real TOD

samples = posterior.sample((5000,), x=mock_tod)

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
plt.show()