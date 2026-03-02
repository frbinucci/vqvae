from __future__ import annotations

import argparse
from sklearn.linear_model import LinearRegression

import numpy as np
import torch

import utils
from gib_quantization.gib_utils import gib_fit
from gib_quantization.quantizer import fit_scalar_lloyd_codebooks, quantize_components
from gib_quantization.rate_allocator import GaussianIBRateAllocatorMinimumRate, EntropyBasedScalarRateAllocator, \
    ReverseWaterFillingRateAllocator
import matplotlib.pyplot as plt

from regression_test import nmse_db


from matplotlib import rcParams

from utils import make_gaussian_hard_case_for_gib


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
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--use_closed_form", action="store_true")  # se vuoi anche OLS
    parser.add_argument("--quantization_bits",type=int,default=35)
    args = parser.parse_args()

    training_data, validation_data, training_loader, validation_loader, _ = \
        utils.load_data_and_data_loaders(args.dataset, args.batch_size,nx=args.nx,ny=args.ny)

    Xtr, Ytr = collect_xy(training_loader, device="cpu")
    Xtest, Ytest = collect_xy(validation_loader, device="cpu")

    #Xtr,Ytr,Xtest,Ytest,_ = make_gaussian_hard_case_for_gib(y_dim=args.ny,n_features=args.nx)

    n_components_array = np.array([5,10,20,24])
    nmse_db_array_inv_wf = np.zeros(len(n_components_array))
    nmse_db_array_gib_aided_quantization = np.zeros(len(n_components_array))

    component_index = 0
    bits_per_mode_wf = np.zeros(len(n_components_array))
    bits_per_mode_gib_aided = np.zeros(len(n_components_array))
    for nz in n_components_array:

        #Part 1 - Inverse Waterfilling quantization
        classical_quantizer = ReverseWaterFillingRateAllocator(args.quantization_bits)
        A,V,_ = gib_fit(Xtr,Ytr,n_comp=nz)
        A = V.T
        Z_tr_inv_wf = (A @ Xtr.T).T
        Z_test_inv_wf = (A @ Xtest.T).T

        classical_quantizer.solve_allocation_problem(Z_tr_inv_wf,nz)
        BitsPerMode = classical_quantizer.optim["BitsPerMode"]
        bits_per_mode_wf[component_index] = args.quantization_bits/np.sum(BitsPerMode>0)

        codebooks,thresholds = fit_scalar_lloyd_codebooks(Z_tr_inv_wf,BitsPerMode)
        idx,Z_q_inv_wf = quantize_components(Z_test_inv_wf,codebooks,thresholds)

        #Part 2 - GIB-aided Quantization
        quantizer = GaussianIBRateAllocatorMinimumRate(args.quantization_bits)
        quantizer.solve_allocation_problem(Xtr,Ytr,nz)
        A_matrix = quantizer.optim["AMatrix"]

        RatePerMode = quantizer.optim["RatePerMode"]
        BitsPerMode = quantizer.optim["BitsPerMode"]
        DistortionPerMode = quantizer.optim["DistortionPerMode"]

        Ztr = A_matrix @ Xtr.T
        Ztr = Ztr.T
        Z_test = A_matrix @ Xtest.T
        Z_test = Z_test.T

        BitsPerMode = quantizer.optim["BitsPerMode"]
        codebooks,thresholds = fit_scalar_lloyd_codebooks(Ztr,BitsPerMode)
        idx,Z_q = quantize_components(Z_test,codebooks,thresholds)

        #L-MMSE estimator computation
        Y_hat_classic_inv_wf = LinearRegression(fit_intercept=True).fit(Z_tr_inv_wf,Ytr).predict(Z_q_inv_wf)
        #Classic quantization NMSE computation
        nmse_db_array_inv_wf[component_index] = 10*np.log10(compute_nmse(Ytest,Y_hat_classic_inv_wf))

        #GIB-AIDED quantization NMSE computation
        Y_hat = LinearRegression(fit_intercept=True).fit(Ztr,Ytr).predict(Z_q)
        nmse_db_array_gib_aided_quantization[component_index] = 10*np.log10(compute_nmse(Ytest,Y_hat))

        bits_per_mode_gib_aided[component_index] = args.quantization_bits/np.sum(BitsPerMode>0)
        component_index+=1

    set_matplotlib_style(14)
    plt.title(rf"Estimation Results GIB Quantization ($N_b={args.quantization_bits}$ bits)")
    plt.plot((n_components_array/args.nx)*100,nmse_db_array_inv_wf,marker='v',label=r'Reverse WF')
    plt.plot((n_components_array/args.nx)*100,nmse_db_array_gib_aided_quantization,marker='o',label=fr'GIB-Aided')
    plt.legend()
    plt.xlabel(f'[%] of components (out of {args.nx})')
    plt.ylabel('NMSE [dB]')
    plt.show()









if __name__=="__main__":
    main()



