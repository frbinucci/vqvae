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
    mu = model(x, deterministic=deterministic, skip_quantization=skip_quantization)[1]
    #mu, _ = out.chunk(2, dim=-1)  # mu: [n_plot, ny]

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


import numpy as np

def make_gaussian_hard_case_for_gib(
    n_samples=int(1e5),
    n_features=25,
    y_dim=3,
    k_strong=4,          # fattori molto rilevanti per Y
    m_weak=8,            # fattori debolmente rilevanti ma ad alta varianza
    scale_strong=0.6,    # std del blocco "forte"
    scale_weak=6,      # std del blocco "debole" (molto più grande!)
    scale_noise=1.0,     # std del rumore irrilevante su X
    gamma_weak=0.15,     # quanto i fattori W entrano in Y (piccolo ma NON zero)
    snr_y=5.0,          # SNR di Y rispetto alla parte pulita (più alto = Y più predicibile)
    rotate=True,
    train_frac=0.8,
    shuffle=True,
    seed=0,
):
    """
    Restituisce: Xtr, Ytr, Xte, Yte, info
    con X gaussiane e Y multidimensionale (joint Gaussian).

    Shapes:
      Xtr: (n_tr, n_features)
      Ytr: (n_tr, y_dim)
      Xte: (n_te, n_features)
      Yte: (n_te, y_dim)
    """
    rng = np.random.default_rng(seed)

    n = int(n_features)
    k = int(k_strong)
    m = int(m_weak)
    d = int(y_dim)
    r = n - k - m
    if r < 0:
        raise ValueError("Serve n_features >= k_strong + m_weak.")

    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac deve essere tra 0 e 1.")

    # Latenti gaussiani
    U = rng.standard_normal((n_samples, k))
    W = rng.standard_normal((n_samples, m))
    V = rng.standard_normal((n_samples, r)) if r > 0 else None

    # X prima della rotazione: [forte, debole, irrilevante]
    parts = [scale_strong * U, scale_weak * W]
    if r > 0:
        parts.append(scale_noise * V)
    X0 = np.concatenate(parts, axis=1)  # (n_samples, n_features)

    # Rotazione ortogonale per mischiare le dimensioni (mantiene gaussianità)
    Q = None
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        X = X0 @ Q
    else:
        X = X0

    # Y multidimensionale: forte da U + debole da W + rumore
    Bu = rng.standard_normal((k, d))
    Bw = rng.standard_normal((m, d))

    Y_clean = (U @ Bu) + gamma_weak * (W @ Bw)  # (n_samples, y_dim)

    # Rumore su Y fissato da snr_y (per-dimensione, cov diagonale)
    var_clean = np.var(Y_clean, axis=0, ddof=1)           # (y_dim,)
    var_noise = var_clean / float(snr_y)
    noise = rng.standard_normal((n_samples, d)) * np.sqrt(var_noise + 1e-18)
    Y = Y_clean + noise

    # Split train/test
    idx = np.arange(n_samples)
    if shuffle:
        rng.shuffle(idx)

    n_tr = int(np.floor(train_frac * n_samples))
    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr:]

    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]

    info = {
        "Q": Q,
        "Bu": Bu,
        "Bw": Bw,
        "k_strong": k,
        "m_weak": m,
        "scale_strong": scale_strong,
        "scale_weak": scale_weak,
        "scale_noise": scale_noise,
        "gamma_weak": gamma_weak,
        "snr_y": snr_y,
        "rotate": rotate,
        "train_frac": train_frac,
        "seed": seed,
    }
    return Xtr, Ytr, Xte, Yte, info



def load_gaussian_xy(**kwargs):
    seed = kwargs.get("seed", 1)
    rng = np.random.default_rng(seed)

    n_samples = kwargs.get("n_samples", int(1e5))
    train_ratio = kwargs.get("train_ratio", 0.8)
    nx = kwargs.get("nx", 10)
    ny = kwargs.get("ny", 5)
    dtype = kwargs.get("dtype", np.float32)

    nmse_db_target = kwargs.get("nmse_db_target", -20)  # dB
    rank = int(min(kwargs.get("rank", min(nx, ny)), nx, ny))
    if rank <= 0:
        raise ValueError("rank deve essere >= 1")

    # X ~ N(0, I)
    X = rng.standard_normal((n_samples, nx)).astype(dtype, copy=False)

    # A with controlled rank (ny x nx)
    U = rng.standard_normal((ny, rank)).astype(dtype, copy=False)
    V = rng.standard_normal((rank, nx)).astype(dtype, copy=False)
    A = (U @ V) / math.sqrt(rank)  # (ny x nx)

    # clean signal
    Y0 = X @ A.T  # (N, ny)

    # choose sigma^2 to hit global NMSE target
    nmse_target = 10.0 ** (nmse_db_target / 10.0)

    sig_power = np.mean(np.sum(Y0 * Y0, axis=1)).item()  # E||Y0||^2

    # Want: E||eps||^2 / E||Y0+eps||^2 = nmse_target
    # E||eps||^2 = ny * sigma^2
    denom = max(1e-12, (1.0 - nmse_target))
    sigma2 = (nmse_target / denom) * (sig_power / ny)

    eps = rng.standard_normal(Y0.shape).astype(dtype, copy=False) * math.sqrt(sigma2)
    Y = Y0 + eps

    # split
    n_train = int(train_ratio * n_samples)
    x_train = X[:n_train, :]
    y_train = Y[:n_train, :]
    x_test  = X[n_train:, :]
    y_test  = Y[n_train:, :]

    # ---- TRUE (population) covariance matrices from the generative model ----
    I_x = np.eye(nx, dtype=dtype)
    I_y = np.eye(ny, dtype=dtype)

    cov_xx_true = I_x
    cov_yy_true = (A @ A.T).astype(dtype, copy=False) + (sigma2 * I_y)
    cov_xy_true = A.T.astype(dtype, copy=False)   # shape (nx, ny)
    cov_yx_true = A.astype(dtype, copy=False)     # shape (ny, nx)

    cov_joint_true = np.block([
        [cov_xx_true, cov_xy_true],
        [cov_yx_true, cov_yy_true],
    ]).astype(dtype, copy=False)

    cov_true = {
        "cov_xx": cov_xx_true,
        "cov_yy": cov_yy_true,
        "cov_xy": cov_xy_true,
        "cov_yx": cov_yx_true,
        "cov_joint": cov_joint_true,
        "A": A.astype(dtype, copy=False),
        "sigma2": float(sigma2),
    }

    #return x_train, y_train, x_test, y_test, cov_true
    print(nx)
    training_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_set = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    return training_set, test_set#, torch.trace(torch.eye(nx, dtype=dtype))  # o altro se ti serve

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
        training_data, validation_data = load_gaussian_xy(nx = nx,ny = ny)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        #x_train_var =y_var

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader#, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()

def fingerprint(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_model_and_results(model, results, hyperparameters, timestamp,output_dir):
    SAVE_MODEL_PATH = os.getcwd() + '/results'+output_dir

    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH,exist_ok=True)

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data'+'.pth')
