import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_posterior(posterior, x_obs, true_value=None, num_samples=5000, title="Posterior"):
    samples = posterior.sample((num_samples,), x=x_obs).numpy().flatten()
    
    kde = gaussian_kde(samples)

    x_plot = np.linspace(samples.min(), samples.max(), 200)
    pdf = kde(x_plot)

    median = np.median(samples)
    std = np.std(samples)

    plt.figure(figsize=(6,4))
    plt.hist(samples, bins=30, density=True, alpha=0.3)
    plt.plot(x_plot, pdf)

    plt.axvline(median, linestyle="--", label=f"Median = {median:.2e}")

    if true_value:
        plt.axvline(true_value, color="red", label=f"True = {true_value:.2e}")

    plt.title(title)
    plt.legend()
    plt.show()


def plot_sbc(z_values):
    plt.figure(figsize=(5,4))
    plt.hist(z_values, bins=20, density=True, alpha=0.7)
    plt.axhline(1.0, color="red", linestyle="--")
    plt.title("SBC check (should be uniform)")
    plt.show()

    
def plot_coverage(z_values):
    z_sorted = np.sort(z_values)
    N = len(z_sorted)
    
    empirical_cdf = np.arange(1, N+1) / N

    plt.plot(z_sorted, empirical_cdf, label="empirical")
    plt.plot([0,1], [0,1], "--", label="ideal")
    
    plt.xlabel("Nominal coverage (z)")
    plt.ylabel("Empirical coverage")
    plt.title("z–z plot (coverage test)")
    plt.legend()
    plt.show()