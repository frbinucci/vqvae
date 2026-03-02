
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack

class TokenizedGaussianIBEncoder(nn.Module):
    """
    Deterministic encoder producing M tokens of dimension D:
      x: [B, in_dim]  ->  z_e: [B, M, D]
    """
    def __init__(self, in_dim: int, n_tokens: int, token_dim: int, init_log_var: float = -2.0):
        super().__init__()
        self.in_dim = in_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.h_dim = n_tokens * token_dim

        # mean: linear -> produces M*D numbers
        self.mu = nn.Linear(in_dim, self.h_dim, bias=True)

        # (optional) diag covariance, constant learnable. Not used in VQ-VAE deterministic mode.
        self.log_var = nn.Parameter(torch.ones(self.h_dim) * init_log_var)


    def forward(self, x, sample: bool = False):
        mu = self.mu(x)  # [B, M*D]
        log_var = self.log_var.unsqueeze(0).expand_as(mu)  # [B, M*D]

        if sample:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std  # [B, M*D]
        else:
            z = mu  # deterministic, VQ-VAE-style

        # tokenize: [B, M, D]
        z_tokens = z.view(z.size(0), self.n_tokens, self.token_dim)
        mu_tokens = mu.view(mu.size(0), self.n_tokens, self.token_dim)
        log_var_tokens = log_var.view(log_var.size(0), self.n_tokens, self.token_dim)

        return mu_tokens, log_var_tokens, z_tokens

class GaussianIBEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, init_log_var=-2.0):
        super().__init__()
        # mean: linear
        self.mu = nn.Linear(in_dim, h_dim, bias=True)

        # diag covariance (constant, learnable): log_var is a parameter vector
        self.log_var = nn.Parameter(torch.ones(h_dim) * init_log_var)

    def forward(self, x, sample=False):

        mu = self.mu(x)  # [B, h_dim]
        log_var = self.log_var.unsqueeze(0).expand_as(mu)  # [B, h_dim]

        if not sample:
            return mu, log_var, mu  # z = mu (no sampling)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, log_var, z


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

    def forward(self, x):
        return self.conv_stack(x)

if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)

