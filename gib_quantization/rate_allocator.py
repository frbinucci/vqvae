import numpy as np

from gib_quantization.gib_utils import gib_fit

class GaussianIBRateAllocator:

    def __init__(self,**kwargs):
        self.rate_tol = kwargs.get('rate_tol',1e-3)
        self.iteration_tol = kwargs.get('iteration_tol',1e-6)
        self.max_iter = kwargs.get('max_iter',100)
        self.max_bisect_iter = kwargs.get('max_bisect_iter',100)
        self.bisect_tol = kwargs.get('bisect_tol',1e-3)
        self.D_min = kwargs.get('D_min',1e-15)
        self.num_tol = kwargs.get('num_tol',1e-14)
        self.max_grow = kwargs.get('max_grow',60)
        self.R_tot = kwargs.get('R_tot',32)
        self.rEps = kwargs.get('rEps',1e-8)
        self.alpha_damp = kwargs.get('alpha_damp',0.7)
        self.lambda_floor = kwargs.get('lambda_floor',1e-12)
        self.sMax = kwargs.get('sMax',1e6)
        self.root_tol = kwargs.get('root_tol',1e-12)
        self.optim = {}


    def solve_allocation_problem(self,X,Y,nz):

        A0,V,info = gib_fit(X,Y,nz)

        beta = info['beta_used']
        lam = np.maximum(info['eigenvalues'],self.lambda_floor)
        s = (info['alpha'])**2
        s = s[0:nz]

        D_over_time = np.zeros((nz,self.max_iter))
        S_over_time = np.zeros((nz,self.max_iter))

        s_prev = s
        D_prev = self.D_min*np.ones(nz)

        for t in range(self.max_iter):
            #Q-step + Eta-step
            R0 = self._compute_rate(lam,s,0,beta)
            if np.abs(R0-self.R_tot)<self.rate_tol:
                eta_star = 0
            else:
                if R0>self.R_tot:
                #Case 1: For \eta=0 the rate exceeds the maximum budget, penalization is needed
                    eta_hi = 1
                    R_hi = self._compute_rate(lam,s,eta_hi,beta)
                    grow = 0
                    while (R_hi>self.R_tot and grow<self.max_grow):
                        eta_hi = eta_hi*2
                        R_hi = self._compute_rate(lam, s, eta_hi,beta)
                        grow = grow+1
                    eta_lo = 0
                else:
                    # Case 2: For \eta=0 the rate is lower than the maximum budget, we need to increase the total rate
                    eta_lo =-1
                    R_lo = self._compute_rate(lam,s,eta_lo,beta)
                    grow=0
                    while (R_lo<self.R_tot and grow<self.max_grow):
                        eta_lo = eta_lo*2
                        R_lo = self._compute_rate(lam,s,eta_lo,beta)
                        grow = grow+1
                    eta_hi = 0
                eta_star = self._compute_eta(lam,s,eta_lo,eta_hi,beta)

            D = self._compute_optimal_distortion(eta_star,s,lam,beta)

            rate_per_mode = self._rate_from_sd(s, D)
            active_modes = rate_per_mode > self.rEps
            Rsum = float(np.sum(rate_per_mode))

            # --- Early Stopping criterion ---
            if (t > 0) and (s_prev is not None):
                relDs = np.linalg.norm(s - s_prev) / max(1e-12, np.linalg.norm(s_prev))
                relDD = np.linalg.norm(D - D_prev) / max(1e-12, np.linalg.norm(D_prev))
                if max(relDs, relDD) < self.iteration_tol and abs(Rsum - self.R_tot) < self.iteration_tol:
                    break

            # Logging
            D_over_time[:, t] = D
            S_over_time[:, t] = s

            s_prev = s.copy()
            D_prev = D.copy()

            # S update
            S_new = self._compute_optimal_s(D, eta_star, lam, beta, active_modes)
            s = (1 - self.alpha_damp) * s + self.alpha_damp * S_new

        rate_per_mode = self._rate_from_sd(s,D)

        self.optim = {
            "AMatrix": np.diag(np.sqrt(s)) @ V.T,
            "RatePerMode": rate_per_mode,
            "BitsPerMode": self._rates_to_integer_bits(rate_per_mode),
            "DistortionPerMode": D
        }

        return s,D,rate_per_mode

    def _compute_optimal_s(self,D,eta,lam,beta,active_modes):
        S = np.zeros(len(active_modes))
        for i in range(len(active_modes)):
            lambda_i = lam[i]
            if active_modes[i]==False:
                S[i] = 0
            else:
                a0 = eta*D[i]**2
                a1 = D[i]*(1-beta*(1-lambda_i)+eta*(1+lambda_i))
                a2 = lambda_i*(1+eta)

                if abs(a2)<self.num_tol:
                    if abs(a1)<self.num_tol:
                        S[i] = D[i]
                    else:
                        S[i] = -a0/a1
                else:
                    delta = a1**2-4*a2*a0

                    cand = np.array([])
                    if delta>=0 and delta!=np.inf:
                        r1 = (-a1-np.sqrt(delta))/(2*a2)
                        r2 = (-a1+np.sqrt(delta))/(2*a2)
                        cand = np.array([r1, r2])
                        cand = cand[np.isfinite(cand) & (cand > 0)]

                    cand = np.asarray(cand).ravel()  # per sicurezza

                    if cand.size == 0:
                        S[i] = D[i]
                    else:
                        # prefer the root >= D (keeps feasibility with D<=s)
                        cand2 = cand[cand >= D[i]]
                        if cand2.size != 0:
                            S[i] = cand2.min()
                        else:
                            S[i] = cand.max()  # last resort

                    S[i] = min(max(S[i], D[i]*(1+self.num_tol)), self.sMax)
        return S


    def _compute_eta(self,lam, s, eta_lo, eta_hi,beta):
        Rlo = self._compute_rate(lam,s,eta_lo,beta)
        Rhi = self._compute_rate(lam,s,eta_hi,beta)

        if (Rlo < self.R_tot) and (Rhi < self.R_tot):
            return eta_lo
        if (Rlo > self.R_tot) and (Rhi > self.R_tot):
            return eta_hi

        for _ in range(self.max_bisect_iter):
            eta_mid = 0.5 * (eta_lo + eta_hi)
            Rmid = self._compute_rate(lam,s,eta_mid,beta)

            if abs(Rmid - self.R_tot) <= self.bisect_tol:
                return eta_mid

            if Rmid > self.R_tot:
                eta_lo = eta_mid
            else:
                eta_hi = eta_mid

        return 0.5 * (eta_lo + eta_hi)

    def _compute_optimal_distortion(self,eta,s,lam,beta):

        nz = len(s)
        D = np.zeros(nz)
        for i in range(len(s)):
            si = s[i]
            lambda_i = lam[i]

            if eta==0:
                denom = beta*(1-lambda_i)-1
                if denom>0:
                    bestDi = (lambda_i*si)/denom
                else:
                    bestDi = si

                bestDi = self._clamp(bestDi,si)
            else:
                a2 = eta
                a1 = si*(1+eta*(1+lambda_i)-beta*(1-lambda_i))
                a0 = (1+eta)*lambda_i*si**2

                delta = a1**2-4*a2*a0
                candidate_solutions = [self.D_min,si]

                if(delta>=0 and delta!=np.inf):
                    r1 = (-a1+np.sqrt(delta))/(2*a2)
                    r2 = (-a1-np.sqrt(delta))/(2*a2)
                    candidate_solutions.append(r1)
                    candidate_solutions.append(r2)

                best_cost = np.inf
                bestDi = si
                for candidate_distortion in candidate_solutions:
                    Dc = self._clamp(candidate_distortion, si)
                    cost = self._evaluate_lagrangian_per_mode(Dc,si,eta,lambda_i,beta)
                    if cost < best_cost:
                        best_cost = cost
                        bestDi = Dc

            bestDi = self._clamp(bestDi, si)
            D[i] = bestDi

        return D


    def _compute_rate(self,lam,s,eta,beta):
        D = self._compute_optimal_distortion(eta , s,lam,beta)
        R = np.sum(self._rate_from_sd(s,D))
        return R

    def _rate_from_sd(self,s, D):

        s = np.asarray(s)
        D = np.asarray(D)

        r = np.zeros_like(D, dtype=float)
        active = (s > 0) & (D > 0) & (D < s)
        r[active] = 0.5 * np.log2(s[active] / D[active])
        return r

    def _evaluate_lagrangian_per_mode(self,Di,si,eta,lambda_i,beta):

        IZX = 0.5 * np.log2((si + Di) / Di)
        IZY = 0.5 * np.log2((si + Di) / (lambda_i * si + Di))
        J = IZX - beta * IZY

        Ri = np.where(Di < si * (1 - self.root_tol), 0.5 * np.log2(si / Di), 0.0)
        Li = J + eta * Ri
        return Li

    def _clamp(self,D,s):
        if np.isinf(D):
            Dc = s
        else:
            Dc = max(D,self.D_min)
            if Dc>s:
                Dc = s
        return Dc

    def _rates_to_integer_bits(self,r):
        r = np.maximum(r, 0.0)
        b = np.floor(r).astype(int)
        if self.R_tot is None:
            return b
        # assegna i bit rimanenti ai maggiori residui frazionari
        rem = int(self.R_tot - b.sum())
        if rem > 0:
            frac = r - np.floor(r)
            idx = np.argsort(-frac)[:rem]
            b[idx] += 1
        return b






