import torch

def load_dataset(theta_path, x_path):
    theta = torch.load(theta_path)
    x = torch.load(x_path)
    x = x.view(x.size(0), -1)
    return theta, x


def train_val_split(theta, x, train_fraction=0.9):
    N = len(theta)
    perm = torch.randperm(N)

    n_train = int(train_fraction * N)

    return (
        theta[perm[:n_train]],
        x[perm[:n_train]],
        theta[perm[n_train:]],
        x[perm[n_train:]],
    )

def normalize(x_train, x_val):
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)

    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    return x_train_n, x_val_n, {"mean": mean, "std": std}