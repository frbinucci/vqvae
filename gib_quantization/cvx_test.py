import cvxpy as cp
import numpy as np
import dccp

import utils
from gib_quantization.gib_utils import gib_fit
from gib_quantization.test import collect_xy
import matplotlib.pyplot as plt


n = 10
beta = 100.0
ln2 = np.log(2)
eps = 1e-9

rho_min = 1          # much safer numerically than 1e-9
rho_max = 1e6           # pick something physically sensible for your SNR scale
P = 1e6                 # optional total budget

rho = cp.Variable(n)
t = cp.Variable()  # scalar

lam = cp.Parameter(n, nonneg=True)   # your lambda_i (given)
R_tot = cp.Parameter(nonneg=True)

training_data, validation_data, training_loader, validation_loader, _ = \
    utils.load_data_and_data_loaders("MULTIVARIATE_GAUSSIAN", 512, nx=20, ny=15)

Xtr, Ytr = collect_xy(training_loader, device="cpu")
Xtest, Ytest = collect_xy(validation_loader, device="cpu")
A, V, info = gib_fit(Xtr, Ytr, n_comp=10)

beta = info['beta_used']
print(beta)

# ---- Known-curvature pieces ----
# f is convex because it's negative * concave
f = -(beta - 1.0) * cp.sum(cp.log1p(rho)) / ln2          # convex

# g is convex because it's negative * concave
g = -beta * cp.sum(cp.log1p(cp.multiply(lam, rho))) / ln2  # convex

# Original objective = f - g
# DCCP-friendly form: minimize f - t  subject to g >= t
objective = cp.Minimize(f-t)  # convex - affine => convex

sum_log2_rho = cp.sum(cp.log(rho)) / ln2  # concave

constraints = [
    rho >= rho_min,
    rho <= rho_max,         # <- key stabilizer
    # or: cp.sum(rho) <= P, # <- also good (can use both)
    sum_log2_rho >= 0,
    sum_log2_rho <= R_tot,
    g >= t,
    t<=0,
]

prob = cp.Problem(objective, constraints)

print("objective curvature:", prob.objective.expr.curvature)
print("is_dccp:", dccp.is_dccp(prob))

# Example parameter values
lambdas = info['eigenvalues']
lam.value = lambdas[0:10]



R_tot.value = 22

# Initialization helps a lot
rho.value = np.ones(n) * 0.5
t.value = float(g.value) - 1e-3



val = prob.solve(method="dccp", solver=cp.MOSEK, verbose=False)
print("value:", val)
print("rho*:", rho.value)
print("CVX Rate: ",np.sum(np.log2(rho.value)))


#####
print("Handmade Solution")

rho_handmade = np.zeros(n)

for i in range(n):
    rho_handmade[i]=max((beta*(1-lambdas[i])-1)/lambdas[i],rho_min)
    rho_handmade[i]=min(rho_handmade[i],rho_max)


R_tot_prov = np.sum(np.log2(rho_handmade))
eta_lo = 5e-5
eta_hi = 5e-1
bisection_rounds = 10000
tol = 1e-4
R_tot_constraint = 22
eta_mid = 0
if R_tot_prov>R_tot_constraint:
    for round in range(bisection_rounds):
        print(round)
        rho_handmade = np.zeros(n)
        eta_mid = np.sqrt(eta_lo * eta_hi)
        for i in range(n):
            Bi = eta_mid-(beta-1)+lambdas[i]*(eta_mid+beta)
            Ai = lambdas[i]*(1+eta_mid)
            Deltai = Bi**2-4*Ai*eta_mid
            if Deltai>0:
                rho_i = (-Bi+np.sqrt(Deltai))/(2*Ai)
                rho_i = max((rho_i),rho_min)
                rho_i = min(rho_i,rho_max)
                rho_handmade[i] = rho_i
            else:
                rho_handmade[i] = rho_min
        r_tot = np.sum(np.log2(rho_handmade))
        R_mid = r_tot - R_tot_constraint
        if abs(R_mid) <= tol:
            eta_star = eta_mid
            break

        if R_mid > 0:  # too much rate -> increase eta
            eta_lo = eta_mid
        else:  # too little rate -> decrease eta
            eta_hi = eta_mid
else:
    eta_star = eta_mid

print(f'Adjusted Rate = {np.sum(np.log2(rho_handmade))}')









plt.plot(rho_handmade)
plt.plot(rho.value)
plt.show()
print(f'Handcrafted solution: {rho_handmade}')
print(f'Handcrafted rate: {np.sum(np.log2(rho_handmade))}')



