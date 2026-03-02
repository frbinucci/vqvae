import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.decoder import gaussian_nll
from models.vqvae import VQVAE
import matplotlib.pyplot as plt
import hashlib



from regression_test import eval_nmse

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

#Training parameters
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_updates", type=int, default=15000)

#D parameter (dimension of the latent space)
parser.add_argument("--token_dim", type=int, default=4)
#Number of tokens
parser.add_argument("--n_token", type=int, default=10)
#Quantization bits
parser.add_argument("--n_embeddings", type=int, default=512)

parser.add_argument("--n_residual_hiddens", type=int, default=16)
parser.add_argument("--n_residual_layers", type=int, default=2)

parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--bn_warmup", type=int, default=5000)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--skip_vector_quantization", action="store_true",default=False)
parser.add_argument("--log_interval", type=int, default=500)


#Bottleneck Loss
parser.add_argument("--ema_decay",type=float,default=0.995)
parser.add_argument("--k_target", type=int, default=32)
parser.add_argument("--beta_bn",type=float,default=0.03)

#Dataset management (Multivariate Normal Regression)
parser.add_argument("--dataset",  type=str, default='MULTIVARIATE_GAUSSIAN')
parser.add_argument("--nx",type=int,default=100)
parser.add_argument("--ny",type=int,default=70)
parser.add_argument("--dataset_size",type=int,default=int(1e4))
parser.add_argument("--train_ratio",type=float,default=0.8)

# whether or not to save model
parser.add_argument("-save", action="store_true",default=True)
parser.add_argument("--filename",  type=str, default=timestamp)

#Test
parser.add_argument("--mode",type=str,default='test')
parser.add_argument("--path_to_load",type=str,default='results/vqvae_data_mon_mar_2_16_47_22_2026.pth')
parser.add_argument("--plot_examples", action="store_true")
parser.add_argument("--n_plot", type=int, default=1)
parser.add_argument("--plot_seed", type=int, default=512)


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size,nx=args.nx,ny=args.ny)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

quantizer_type = 'convolutional'
if args.dataset=='MULTIVARIATE_GAUSSIAN':
    quantizer_type='shallow'
    model = VQVAE(args.n_token, args.n_residual_hiddens,
                  args.n_residual_layers, args.n_embeddings, args.token_dim, args.beta,quantizer_type=quantizer_type,in_dim=args.nx,out_dim=args.ny,skip_quantization=args.skip_vector_quantization).to(device)
else:
    model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
                  args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)



"""
Set up optimizer and training loop
"""
if args.mode=='train':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    model.train()

    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }
#elif args.mode=='test':
#    model.load_state_dict(torch.load(args.path_to_load,weights_only=False)['model'])
#    model.eval()


def entropy_from_counts(counts, eps=1e-12):
    p = counts / (counts.sum() + eps)
    return -(p * (p + eps).log()).sum()

# maintain EMA of code usage across steps
ema_counts = torch.zeros(args.n_embeddings, device=device)

def train():
    output_dir = f'/compressed_size_{int(args.n_token * args.token_dim)}_bits_{int(args.n_token * np.log2(args.n_embeddings))}/'

    print(f'Checkpoints will be saved at: {output_dir}')
    global ema_counts
    data_iter = iter(training_loader)

    for i in range(args.n_updates):
        model.train()

        beta_bn = args.beta_bn * min(1.0, i / float(args.bn_warmup))

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(training_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)   # <-- FIX
        optimizer.zero_grad()


        embedding_loss, out, perplexity, code_idx,H_soft,log_var = model(x)  # code_idx: [B]
        #y_mu, y_logvar = out.chunk(2, dim=-1)

        H_target = math.log(args.k_target)  # nats, es: k_target=32,64,128...
        loss_fn = nn.MSELoss()
        recon_loss = gaussian_nll(y,out,log_var)

        if args.skip_vector_quantization==False:
            code_idx = code_idx.view(-1).long()  # [B]
            batch_counts = torch.bincount(code_idx.cpu(), minlength=args.n_embeddings).float().to(device)

            ema_counts = args.ema_decay * ema_counts + (1 - args.ema_decay) * batch_counts
            loss = recon_loss + args.beta * embedding_loss + beta_bn * (H_soft - H_target).pow(2)

        else:
            H_soft=0
            H_target=0
            loss = recon_loss + beta_bn * (H_soft - H_target) ** 2

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.detach().cpu().numpy())
        results["perplexities"].append(perplexity.detach().cpu().numpy())
        results["loss_vals"].append(loss.detach().cpu().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:

            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(model, results, hyperparameters, args.filename,output_dir)
            va = eval_nmse(model, validation_loader, device)
            if args.skip_vector_quantization==False:
                print('Update #', i,
                      'Recon Error:', np.mean(results["recon_errors"][-args.log_interval:]),
                      'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                      'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]),
                      'Unique codes (batch):', code_idx.unique().numel(),
                      'Validation NMSE (db):', va)
            else:
                print('Update #', i,
                      'Recon Error:', np.mean(results["recon_errors"][-args.log_interval:]),
                      'Loss', np.mean(results["loss_vals"][-args.log_interval:]))


if __name__ == "__main__":
    if args.mode == 'train':
        train()
    elif args.mode=='test':

        n_bits_array = np.array([40,50,60,70,80,90])


        nmse_array = np.zeros(len(n_bits_array))
        index=0
        for n in n_bits_array:
            n_embeddings = int(2**int(n/args.n_token))

            model = VQVAE(args.n_token, args.n_residual_hiddens,
                          args.n_residual_layers, n_embeddings, args.token_dim, args.beta,
                          quantizer_type=quantizer_type, in_dim=args.nx, out_dim=args.ny,
                          skip_quantization=args.skip_vector_quantization).to(device)

            ckpt = torch.load(f'results/compressed_size_40_bits_{n}/vqvae_data.pth', map_location=device,weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval()
            va = eval_nmse(model, validation_loader, device)
            nmse_array[index] = va
            index+=1

        nmse_gib_aided = np.load('./res_gib_aided/nmse_array.npy')
        rate_array_gib_aided = np.load('./res_gib_aided/rate_array.npy')

        plt.plot(rate_array_gib_aided[0,0,:],nmse_gib_aided[0,0,:],marker='o',label='GIB-Aided')
        plt.plot(n_bits_array,nmse_array,marker='v',label='VQ-VAE')
        plt.xlabel('Rate [bits]',fontsize=14)
        plt.ylabel('NMSE [dB]',fontsize=14)
        plt.grid()
        plt.legend()
        plt.show()
        print("Validation NMSE (dB):", va)
        utils.plot_random_vectors(model, validation_loader, device,
                                  n_plot=args.n_plot, seed=args.plot_seed,
                                  deterministic=True, skip_quantization=False)
