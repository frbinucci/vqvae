import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TokenVQ(nn.Module):
    """
    Shared-codebook VQ for tokenized latents:
      z_e: [B, M, D]  ->  z_q: [B, M, D]
    Returns:
      vq_loss, z_q_st, perplexity, indices, H_soft
    """
    def __init__(self, n_e: int, e_dim: int, beta: float, init_scale: float = 0.1):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_e, e_dim)

        nn.init.uniform_(self.embedding.weight, -init_scale, init_scale)

    def forward(self, z_e: torch.Tensor, tau: float = 0.5):
        """
        z_e: [B, M, D]
        """


        assert z_e.dim() == 3 and z_e.size(-1) == self.e_dim, f"z_e={z_e.shape}, e_dim={self.e_dim}"
        B, M, D = z_e.shape

        # Flatten tokens: [B*M, D]
        z = z_e.reshape(B * M, D)

        # Compute squared L2 distances to embeddings: [B*M, K]
        # d = ||z||^2 + ||e||^2 - 2 z e^T
        e = self.embedding.weight  # [K, D]
        d = (z.pow(2).sum(1, keepdim=True)
             + e.pow(2).sum(1).unsqueeze(0)
             - 2.0 * z @ e.t())

        # Hard assignment
        indices = torch.argmin(d, dim=1)  # [B*M]
        z_q = self.embedding(indices)     # [B*M, D]

        # Loss terms (Eq. 3)
        codebook_loss   = (z_q - z.detach()).pow(2).mean()
        commitment_loss = self.beta * (z - z_q.detach()).pow(2).mean()
        vq_loss = codebook_loss + commitment_loss

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()  # [B*M, D]
        z_q_st = z_q_st.view(B, M, D)

        # Perplexity (hard)
        enc = F.one_hot(indices, num_classes=self.n_e).type_as(z)  # [B*M, K]
        e_mean = enc.mean(0)  # [K]
        perplexity = torch.exp(-(e_mean * (e_mean + 1e-10).log()).sum())

        # Differentiable "soft" entropy proxy (optional)
        p = torch.softmax(-d / max(tau, 1e-8), dim=1)  # [B*M, K]
        p_mean = p.mean(dim=0)  # [K]
        H_soft = -(p_mean * (p_mean + 1e-10).log()).sum()  # nats

        # Reshape indices to [B, M]
        indices = indices.view(B, M)

        return vq_loss, z_q_st, perplexity, indices, H_soft

class GaussianIBVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        print(self.n_e)
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z):
        # z: [B, D]
        assert z.dim() == 2 and z.size(1) == self.e_dim, f"z={z.shape}, e_dim={self.e_dim}"

        # distances: [B, n_e]
        d = (z.pow(2).sum(1, keepdim=True)
             + self.embedding.weight.pow(2).sum(1)
             - 2 * z @ self.embedding.weight.t())

        indices = torch.argmin(d, dim=1)  # [B] long
        enc = F.one_hot(indices, num_classes=self.n_e).type_as(z)  # [B, n_e]

        z_q = self.embedding(indices)     # [B, D]

        # VQ-VAE standard (ordine corretto)
        codebook_loss   = (z_q - z.detach()).pow(2).mean()
        commitment_loss = self.beta * (z - z_q.detach()).pow(2).mean()
        loss = codebook_loss + commitment_loss

        # straight-through
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = enc.mean(0)
        perplexity = torch.exp(-(e_mean * (e_mean + 1e-10).log()).sum())

        tau = 0.5
        p = torch.softmax(-d / tau, dim=1)  # tau > 0 (es. 0.5 -> 1.0)
        p_mean = p.mean(dim=0)  # [n_e]
        H_soft = -(p_mean * (p_mean + 1e-10).log()).sum()  # nats, differenziabile

        return loss, z_q, perplexity, enc, indices,H_soft

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
