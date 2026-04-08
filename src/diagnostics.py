import numpy as np

def compute_sbc(posterior, x_val, theta_val, n_samples=1000):
    z_values = []
    medians = []

    for i in range(len(x_val)):
        x_i = x_val[i:i+1]
        theta_true = theta_val[i].item()

        samples = posterior.sample(
            (n_samples,),
            x=x_i,
            show_progress_bars=False
        ).numpy().flatten()

        medians.append(np.median(samples))

        z = np.mean(samples <= theta_true)
        z_values.append(z)

    return np.array(z_values), np.array(medians)

