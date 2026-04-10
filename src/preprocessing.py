import torch

def load_dataset(theta_path, x_path):
    theta = torch.load(theta_path)
    x = torch.load(x_path)
    x = x.view(x.size(0), -1)
    return theta, x


def normalize(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)

    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    x_n = (x - mean) / std

    return x_n, {"mean": mean, "std": std}