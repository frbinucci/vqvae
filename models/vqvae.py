from keyword import kwlist

import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder, GaussianIBEncoder
from models.quantizer import VectorQuantizer, GaussianIBVectorQuantizer
from models.decoder import Decoder, GaussianIBDecoder


class VQVAE(nn.Module,):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, quantizer_type='convolutional',in_dim=3,out_dim=3,save_img_embedding_map=False,skip_quantization=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space

        self.skip_quantization = skip_quantization

        if quantizer_type=='shallow':
            self.encoder = GaussianIBEncoder(in_dim,h_dim)
            self.pre_quantization_conv = nn.Identity()
            self.vector_quantization = GaussianIBVectorQuantizer(n_embeddings, h_dim, beta)  # <-- FIX
            self.decoder = GaussianIBDecoder(out_dim,h_dim)
        elif quantizer_type=='convolutional':
            self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
            self.pre_quantization_conv = nn.Conv2d(
                h_dim, embedding_dim, kernel_size=1, stride=1)
            # pass continuous latent vector through discretization bottleneck
            self.vector_quantization = VectorQuantizer(
                n_embeddings, embedding_dim, beta)
            # decode the discrete latent representation
            self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False,deterministic=True,skip_quantization=False):
        if self.skip_quantization==True:
            mu, log_var, z_e = self.encoder(x)
            embedding_loss = 0
            indices = 0
            H_soft = 0
            out = self.decoder(mu)
            perplexity = 0
        else:
            mu, log_var, z = self.encoder(x)

            z_e = mu if deterministic else z
            z_e = self.pre_quantization_conv(z_e)

            embedding_loss, z_q, perplexity, _, indices,H_soft = self.vector_quantization(z_e)
            out = self.decoder(z_q)



        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', out.shape)
            assert False

        return embedding_loss, out, perplexity, indices,H_soft