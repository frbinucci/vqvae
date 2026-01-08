from __future__ import annotations

import argparse
from sklearn.linear_model import LinearRegression

import numpy as np
import torch

import utils
from gib_quantization.quantizer import fit_scalar_lloyd_codebooks, quantize_components
from gib_quantization.rate_allocator import GaussianIBRateAllocator
import matplotlib.pyplot as plt

from regression_test import nmse_db


from matplotlib import rcParams


def set_matplotlib_style(font_size: int = 14) -> None:
    """Set consistent matplotlib styling for paper-quality figures."""
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.family"] = "STIXGeneral"
    rcParams["font.size"] = font_size
    rcParams["legend.fontsize"] = "medium"
    rcParams["axes.grid"] = True

def compute_nmse(Y,Y_hat):
    nmse = 0
    for item in range(Y.shape[0]):
        nmse+=np.sum((Y[item,:]-Y_hat[item,:])**2)/np.sum(Y[item,:]**2)

    nmse/=Y.shape[0]
    return nmse


def collect_xy(loader, device="cpu"):
    Xs, Ys = [], []
    for x, y in loader:
        Xs.append(x.to(device))
        Ys.append(y.to(device))
    return torch.cat(Xs, dim=0).detach().cpu().numpy(), torch.cat(Ys, dim=0).detach().cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MULTIVARIATE_GAUSSIAN")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=35)
    parser.add_argument("--use_closed_form", action="store_true")  # se vuoi anche OLS
    parser.add_argument("--quantization_bits",type=int,default=32)
    args = parser.parse_args()

    training_data, validation_data, training_loader, validation_loader, _ = \
        utils.load_data_and_data_loaders(args.dataset, args.batch_size,nx=args.nx,ny=args.ny)

    Xtr, Ytr = collect_xy(training_loader, device="cpu")
    Xtest, Ytest = collect_xy(validation_loader, device="cpu")

    n_components_array = np.array([1,5,10,15,20,25])
    nmse_db_array = np.zeros(len(n_components_array))
    nmse_quantization_array = np.zeros(len(n_components_array))

    component_index = 0
    for nz in n_components_array:
        quantizer = GaussianIBRateAllocator(R_tot=args.quantization_bits)
        quantizer.solve_allocation_problem(Xtr,Ytr,nz)

        A_matrix = quantizer.optim["AMatrix"]
        RatePerMode = quantizer.optim["RatePerMode"]
        BitsPerMode = quantizer.optim["BitsPerMode"]
        DistortionPerMode = quantizer.optim["DistortionPerMode"]

        Ztr = A_matrix @ Xtr.T
        Ztr = Ztr.T

        Z_test = A_matrix @ Xtest.T
        Z_test = Z_test.T

        #L-MMSE estimator computation
        reg = LinearRegression(fit_intercept=True).fit(Ztr,Ytr)
        Y_hat = reg.predict(Z_test)

        #Ideal NMSE
        nmse_db_array[component_index] = 10*np.log10(compute_nmse(Ytest,Y_hat))

        #Quantization and Decoding
        codebooks,thresholds = fit_scalar_lloyd_codebooks(Ztr,BitsPerMode)
        idx,Z_q = quantize_components(Z_test,codebooks,thresholds)

        Y_hat_quantized = reg.predict(Z_q)

        nmse_quantization_array[component_index] = 10*np.log10(compute_nmse(Ytest,Y_hat_quantized))

        component_index+=1

    set_matplotlib_style(14)
    plt.plot(n_components_array,nmse_db_array,marker='v',label=r'Ideal')
    plt.plot(n_components_array,nmse_quantization_array,marker='o',label=fr'$N_b={args.quantization_bits}$ bits')
    plt.legend()
    plt.xlabel('Number of Components')
    plt.ylabel('NMSE [dB]')
    plt.show()









if __name__=="__main__":
    main()



