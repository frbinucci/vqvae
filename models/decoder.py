
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack

class GaussianRegressionDecoder(nn.Module):
    """
    Predicts Y given quantized tokens.
      z_q: [B, M, D] -> flatten -> [B, M*D] -> outputs [B, 2*out_dim]
    Interpreted as (mu_y, log_var_y).
    """
    def __init__(self, out_dim: int, n_tokens: int, token_dim: int, hidden: int = None):
        super().__init__()
        in_dim = n_tokens * token_dim
        hidden = hidden or (2 * in_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * out_dim)
        )

    def forward(self, z_q: torch.Tensor):
        B, M, D = z_q.shape
        h = z_q.reshape(B, M * D)
        out = self.net(h)
        mu_y, log_var_y = out.chunk(2, dim=1)
        return mu_y, log_var_y


def gaussian_nll(y, mu, log_var):
    # y, mu, log_var: [B, out_dim]
    # per-dimension Gaussian NLL (up to constant)
    return 0.5 * (log_var + (y - mu).pow(2) / torch.exp(log_var)).mean()

class GaussianIBDecoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, out_dim, h_dim):
        super(GaussianIBDecoder, self).__init__()

        self.decoding_stack = nn.Sequential(
            nn.Linear(h_dim,2*out_dim)
        )

    def forward(self, x):
        return self.decoding_stack(x)

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
