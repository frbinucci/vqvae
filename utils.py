import hashlib
import math
from keyword import kwlist

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import dtype
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
from torch.utils.data import TensorDataset, random_split
import numpy as np

@torch.no_grad()
def eval_nmse(model, loader, device):
    was_training = model.training
    model.eval()
    se, sy, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, out, _, _, _ = model(x)
        mu, _ = out.chunk(2, dim=-1)
        se += (y - mu).pow(2).sum().item()
        sy += y.pow(2).sum().item()
        n += y.size(0)
    if was_training:
        model.train()
    nmse = se / (sy + 1e-12)
    return 10.0 * math.log10(nmse + 1e-12)


import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def plot_random_vectors(model, loader, device, n_plot=5, seed=0,
                        deterministic=True, skip_quantization=False):
    was_training = model.training
    model.eval()

    ds = loader.dataset
    N = len(ds)

    # supporta sia Dataset che Subset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(low=0, high=N, size=(n_plot,), generator=g).tolist()

    xs, ys = [], []
    for i in idx:
        x_i, y_i = ds[i]
        xs.append(x_i.unsqueeze(0))
        ys.append(y_i.unsqueeze(0))

    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)

    # forward (adatta i flag se il tuo forward li supporta)
    out = model(x, deterministic=deterministic, skip_quantization=skip_quantization)[1]
    mu, _ = out.chunk(2, dim=-1)  # mu: [n_plot, ny]

    y_cpu = y.detach().cpu()
    mu_cpu = mu.detach().cpu()

    # ---- Plot 1: vettori (component-wise) ----
    plt.figure()
    for k in range(n_plot):
        plt.plot(y_cpu[k].numpy(), label=f"y true #{k}")
        plt.plot(mu_cpu[k].numpy(), linestyle="--", label=f"y pred #{k}")
    plt.title(f"Random examples: true vs pred (n={n_plot})")
    plt.xlabel("component index")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()

    plt.show()

    if was_training:
        model.train()


def load_gaussian_xy(**kwargs):
    seed = kwargs.get("seed", 0)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)

    n_samples = kwargs.get("n_samples", int(1e5))
    train_ratio = kwargs.get("train_ratio", 0.8)
    nx = kwargs.get("nx", 10)
    ny = kwargs.get("ny", 5)
    dtype = kwargs.get("dtype", torch.float32)

    # controlli principali
    nmse_db_target = kwargs.get("nmse_db_target", -10.0)   # es: -3.5, -10, -20...
    rank = kwargs.get("rank", min(nx, ny))                 # puoi metterlo <= bottleneck (es 10)

    # X ~ N(0, I)
    X = torch.randn(n_samples, nx, generator=g, dtype=dtype)

    # A con rango controllato (ny x nx)
    U = torch.randn(ny, rank, generator=g, dtype=dtype)
    V = torch.randn(rank, nx, generator=g, dtype=dtype)
    A = (U @ V) / math.sqrt(rank)   # scala “ragionevole”

    # segnale pulito
    Y0 = X @ A.t()  # [N, ny]

    # scegli sigma^2 per ottenere NMSE target (globale)
    nmse_target = 10 ** (nmse_db_target / 10.0)  # attenzione: db -> ratio
    sig_power = (Y0.pow(2).sum(dim=1)).mean().item()  # E||Y0||^2
    # Vogliamo: E||eps||^2 / E||Y0+eps||^2 = nmse_target circa
    # Con eps ~ N(0, sigma^2 I): E||eps||^2 = ny * sigma^2
    sigma2 = (nmse_target / max(1e-12, (1.0 - nmse_target))) * (sig_power / ny)

    eps = torch.randn(Y0.shape, generator=g, device=Y0.device, dtype=Y0.dtype) * math.sqrt(sigma2)
    Y = Y0 + eps

    dataset = TensorDataset(X, Y)
    n_train = int(train_ratio * n_samples)
    n_val = n_samples - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    return train_ds, val_ds, torch.trace(torch.eye(nx, dtype=dtype))  # o altro se ti serve

def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size,**kwargs):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    elif dataset=='MULTIVARIATE_GAUSSIAN':
        nx = kwargs.get('nx',None)
        ny = kwargs.get('ny',None)
        training_data, validation_data,y_var = load_gaussian_xy(nx = nx,ny = ny)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var =y_var

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()

def fingerprint(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
