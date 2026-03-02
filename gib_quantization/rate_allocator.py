from abc import abstractmethod

import numpy as np

from gib_quantization.gib_utils import gib_fit, cov, logdet_spd


class RateAllocator:

    def __init__(self,R_tot):
        self.R_tot = R_tot

    @abstractmethod
    def solve_allocation_problem(self,X,Y,nz):
        pass


    def _rates_to_integer_bits(self,r):
        r = np.maximum(r, 1)
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


class EntropyBasedScalarRateAllocator(RateAllocator):
    def __init__(self,R_tot):
        super().__init__(R_tot)

    def solve_allocation_problem(self,X,Y,nz):

        nx = X.shape[1]

        rate_per_mode = np.zeros(nx)


        for i in range(nx):
            rate_per_mode[i] = self._compute_gaussian_entropy(X[:,i])

        self.optim = {'RatePerMode': rate_per_mode,
                      'BitsPerMode': self._rates_to_integer_bits(rate_per_mode)}



    def _compute_gaussian_entropy(self,Z,unit='bits'):
        Szz = cov(np.expand_dims(Z,axis=1))

        delta_ref = self._pick_delta_ref_by_reference_bits(Szz)
        allowed_units = {'bits','nats'}
        unit = unit.strip().lower()

        if unit not in allowed_units:
            raise ValueError(f"Unknown entropy unit! Please, specificy a value among {', '.join(allowed_units)}")

        log_det = (1/2)*logdet_spd(Szz)
        if unit=='bits':
            log_det/=np.log(2)

        return log_det

    def _pick_delta_ref_by_reference_bits(self,sigma2, b0=4):
        sigma = np.sqrt(np.asarray(sigma2, float))
        sigma_g = np.median(sigma)
        return np.sqrt(2 * np.pi * np.e) * sigma_g / (2 ** b0)



import numpy as np

class GaussianIBRateAllocatorMinimumRate(RateAllocator):

    def __init__(self, R_tot, **kwargs):
        super().__init__(R_tot)
        self.rate_tol = kwargs.get('rate_tol', 1e-3)
        self.iteration_tol = kwargs.get('iteration_tol', 1e-6)
        self.max_iter = kwargs.get('max_iter', 100)
        self.max_bisect_iter = kwargs.get('max_bisect_iter', 100)
        self.bisect_tol = kwargs.get('bisect_tol', 1e-3)
        self.D_min = kwargs.get('D_min', 1e-15)
        self.num_tol = kwargs.get('num_tol', 1e-14)
        self.max_grow = kwargs.get('max_grow', 60)
        self.rEps = kwargs.get('rEps', 1e-8)
        self.alpha_damp = kwargs.get('alpha_damp', 0.7)
        self.lambda_floor = kwargs.get('lambda_floor', 1e-12)
        self.sMax = kwargs.get('sMax', 1e6)
        self.root_tol = kwargs.get('root_tol', 1e-12)

        # >>> NUOVO: rate minima per modo
        self.r_min = kwargs.get('r_min', 1.0)

        self.optim = {}

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
    # ------------------------------------------------------------
    # NUOVO: trova D_cap(s) tale che rate(s, D_cap) = r_min
    # (ritorna il più grande D che mantiene rate >= r_min)
    # ------------------------------------------------------------
    def _distortion_cap_from_rate(self, s, r_min):
        s = np.asarray(s, dtype=float)
        nz = s.size

        # Bracket su D: D_low = D_min (rate alta), D_high ~ s (rate bassa)
        D_low = np.full(nz, self.D_min, dtype=float)
        D_high = np.maximum(s, self.D_min * 2.0)

        # Verifica rate massima al minimo D (se < r_min => infeasible)
        r_at_low = self._rate_from_sd(s, D_low)
        if np.any(r_at_low < r_min - 1e-10):
            bad = np.where(r_at_low < r_min - 1e-10)[0]
            raise ValueError(
                f"Infeasible: alcuni modi non raggiungono r_min={r_min} neanche a D_min. "
                f"Indici: {bad.tolist()}, r_max~{r_at_low[bad]}"
            )

        # Assicurati che a D_high la rate sia <= r_min (se no, aumenta D_high)
        r_at_high = self._rate_from_sd(s, D_high)
        grow = 0
        while np.any(r_at_high > r_min + 1e-12) and grow < self.max_grow:
            mask = (r_at_high > r_min + 1e-12)
            D_high[mask] = np.minimum(D_high[mask] * 2.0, self.sMax)
            r_at_high = self._rate_from_sd(s, D_high)
            grow += 1

        if np.any(r_at_high > r_min + 1e-8):
            bad = np.where(r_at_high > r_min + 1e-8)[0]
            raise ValueError(
                f"Infeasible numericamente: non riesco a brackettare D per r_min={r_min} "
                f"su alcuni modi {bad.tolist()}. Controlla monotonicità di _rate_from_sd o sMax."
            )

        # Bisezione vettoriale: vogliamo r(D)=r_min con r decrescente in D
        # Manteniamo invarianti: rate(D_low) >= r_min, rate(D_high) <= r_min
        for _ in range(self.max_bisect_iter):
            D_mid = 0.5 * (D_low + D_high)
            r_mid = self._rate_from_sd(s, D_mid)

            # se r_mid >= r_min => D_mid troppo piccolo => sposta D_low in su
            up = (r_mid >= r_min)
            D_low[up] = D_mid[up]

            # se r_mid < r_min => D_mid troppo grande => sposta D_high in giù
            down = ~up
            D_high[down] = D_mid[down]

            if np.max(np.abs(D_high - D_low)) <= self.root_tol * max(1.0, np.max(D_high)):
                break

        # D_low è "il più grande D" che mantiene rate >= r_min (più sicuro di D_high)
        return np.maximum(D_low, self.D_min)

    # ------------------------------------------------------------
    # NUOVO: rate totale coerente con il vincolo r_i >= r_min
    # ------------------------------------------------------------
    def _Rsum_given_eta_with_floor(self, lam, s, eta, beta, D_cap):
        D = self._compute_optimal_distortion(eta, s, lam, beta)
        D = np.maximum(D, self.D_min)
        D = np.minimum(D, D_cap)  # impone r_i >= r_min
        r = self._rate_from_sd(s, D)
        return float(np.sum(r)), D, r

    # ------------------------------------------------------------
    # NUOVO: bisezione su eta usando la stessa definizione di rate
    # ------------------------------------------------------------
    def _bisect_eta_for_budget(self, lam, s, eta_lo, eta_hi, beta, D_cap):
        # assumiamo R(eta) monotona decrescente in eta
        R_lo, _, _ = self._Rsum_given_eta_with_floor(lam, s, eta_lo, beta, D_cap)
        R_hi, _, _ = self._Rsum_given_eta_with_floor(lam, s, eta_hi, beta, D_cap)

        # Bracket atteso: R_lo >= R_tot >= R_hi
        if not (R_lo + 1e-12 >= self.R_tot >= R_hi - 1e-12):
            # se non bracketta, restituisci l'estremo più vicino
            if abs(R_lo - self.R_tot) < abs(R_hi - self.R_tot):
                return eta_lo
            return eta_hi

        lo, hi = eta_lo, eta_hi
        best = 0.5 * (lo + hi)
        for _ in range(self.max_bisect_iter):
            mid = 0.5 * (lo + hi)
            R_mid, _, _ = self._Rsum_given_eta_with_floor(lam, s, mid, beta, D_cap)
            best = mid

            if abs(R_mid - self.R_tot) <= self.bisect_tol:
                return mid

            if R_mid > self.R_tot:
                # rate troppo alta => aumenta eta (sposta lo verso mid)
                lo = mid
            else:
                # rate troppo bassa => diminuisci eta (sposta hi verso mid)
                hi = mid

            if abs(hi - lo) <= self.root_tol * max(1.0, abs(mid)):
                break

        return best

    def _rate_from_sd(self,s, D):

        s = np.asarray(s)
        D = np.asarray(D)

        r = np.zeros_like(D, dtype=float)
        active = (s > 0) & (D > 0) & (D < s)
        r[active] = 0.5 * np.log2(s[active] / D[active])
        return r
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

    def _clamp(self,D,s):
        if np.isinf(D):
            Dc = s
        else:
            Dc = max(D,self.D_min)
            if Dc>s:
                Dc = s
        return Dc

    def _evaluate_lagrangian_per_mode(self,Di,si,eta,lambda_i,beta):

        IZX = 0.5 * np.log2((si + Di) / Di)
        IZY = 0.5 * np.log2((si + Di) / (lambda_i * si + Di))
        J = IZX - beta * IZY

        Ri = np.where(Di < si * (1 - self.root_tol), 0.5 * np.log2(si / Di), 0.0)
        Li = J + eta * Ri
        return Li

    # ------------------------------------------------------------
    # AGGIORNATO: solve con vincolo r_i >= 1 per ogni modo
    # ------------------------------------------------------------
    def solve_allocation_problem(self, X, Y, nz):
        A0, V, info = gib_fit(X, Y, nz)

        beta = info['beta_used']
        lam = np.maximum(info['eigenvalues'], self.lambda_floor)
        s = (info['alpha'])**2
        s = s[0:nz].astype(float)

        # Fattibilità minima di budget
        if self.R_tot < nz * self.r_min - 1e-12:
            raise ValueError(
                f"Infeasible: R_tot={self.R_tot} < nz*r_min={nz*self.r_min}."
            )

        D_over_time = np.zeros((nz, self.max_iter))
        S_over_time = np.zeros((nz, self.max_iter))

        s_prev = s.copy()
        D_prev = self.D_min * np.ones(nz)

        for t in range(self.max_iter):

            # calcola D_cap per l'attuale s (per imporre r_i >= r_min)
            D_cap = self._distortion_cap_from_rate(s, self.r_min)

            # Rate minima (con floor) e massima (con D_min) per checks robusti
            R_min = nz * self.r_min
            R_max = float(np.sum(self._rate_from_sd(s, self.D_min * np.ones_like(s))))
            if self.R_tot > R_max + 1e-9:
                raise ValueError(
                    f"Infeasible: R_tot={self.R_tot} > R_max~{R_max} con D_min={self.D_min}."
                )

            # Q-step + Eta-step: usa SEMPRE la rate coerente con D_cap
            R0, _, _ = self._Rsum_given_eta_with_floor(lam, s, 0.0, beta, D_cap)

            if abs(R0 - self.R_tot) < self.rate_tol:
                eta_star = 0.0
            else:
                if R0 > self.R_tot:
                    # Serve penalizzazione: cerca eta_hi > 0 tale che R(eta_hi) <= R_tot
                    eta_lo = 0.0
                    eta_hi = 1.0
                    grow = 0
                    R_hi, _, _ = self._Rsum_given_eta_with_floor(lam, s, eta_hi, beta, D_cap)
                    while (R_hi > self.R_tot) and (grow < self.max_grow):
                        eta_hi *= 2.0
                        R_hi, _, _ = self._Rsum_given_eta_with_floor(lam, s, eta_hi, beta, D_cap)
                        grow += 1
                else:
                    # Serve aumentare la rate: cerca eta_lo < 0 tale che R(eta_lo) >= R_tot
                    eta_hi = 0.0
                    eta_lo = -1.0
                    grow = 0
                    R_lo, _, _ = self._Rsum_given_eta_with_floor(lam, s, eta_lo, beta, D_cap)
                    while (R_lo < self.R_tot) and (grow < self.max_grow):
                        eta_lo *= 2.0
                        R_lo, _, _ = self._Rsum_given_eta_with_floor(lam, s, eta_lo, beta, D_cap)
                        grow += 1

                eta_star = self._bisect_eta_for_budget(lam, s, eta_lo, eta_hi, beta, D_cap)

            # Distorsione ottima, poi proiezione sul vincolo r_i >= r_min
            D = self._compute_optimal_distortion(eta_star, s, lam, beta)
            D = np.maximum(D, self.D_min)
            D = np.minimum(D, D_cap)

            rate_per_mode = self._rate_from_sd(s, D)
            Rsum = float(np.sum(rate_per_mode))

            # Early stopping (usa rate_tol qui, più coerente)
            if t > 0:
                relDs = np.linalg.norm(s - s_prev) / max(1e-12, np.linalg.norm(s_prev))
                relDD = np.linalg.norm(D - D_prev) / max(1e-12, np.linalg.norm(D_prev))
                if max(relDs, relDD) < self.iteration_tol and abs(Rsum - self.R_tot) < self.rate_tol:
                    break
                print(relDs)


            D_over_time[:, t] = D
            S_over_time[:, t] = s

            s_prev = s.copy()
            D_prev = D.copy()

            # >>> IMPORTANTE: se vuoi r_i >= 1 per TUTTI, NON disattivare modi
            active_modes = np.ones(nz, dtype=bool)

            # S update
            S_new = self._compute_optimal_s(D, eta_star, lam, beta, active_modes)
            s = (1 - self.alpha_damp) * s + self.alpha_damp * S_new

        # Output finale coerente col floor
        D_cap = self._distortion_cap_from_rate(s, self.r_min)
        _, D, rate_per_mode = self._Rsum_given_eta_with_floor(lam, s, eta_star, beta, D_cap)

        bits = self._rates_to_integer_bits(rate_per_mode)
        # Se vuoi garantire anche la versione intera >= 1:
        bits = np.maximum(bits, 1)

        self.optim = {
            "AMatrix": np.diag(np.sqrt(s)) @ V.T,
            "RatePerMode": rate_per_mode,
            "BitsPerMode": bits,
            "DistortionPerMode": D
        }

        return s, D, rate_per_mode




class GaussianIBRateAllocator(RateAllocator):

    def __init__(self, R_tot,**kwargs):
        super().__init__(R_tot)
        self.rate_tol = kwargs.get('rate_tol',1e-3)
        self.iteration_tol = kwargs.get('iteration_tol',1e-6)
        self.max_iter = kwargs.get('max_iter',30)
        self.max_bisect_iter = kwargs.get('max_bisect_iter',100)
        self.bisect_tol = kwargs.get('bisect_tol',1e-3)
        self.D_min = kwargs.get('D_min',1e-15)
        self.num_tol = kwargs.get('num_tol',1e-14)
        self.max_grow = kwargs.get('max_grow',100)
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

    def Rsum_from_eta(eta):
        D = self._compute_optimal_distortion(eta, s, lam, beta)
        r = self._rate_from_sd(s, D)
        r[r < self.rEps] = 0.0  # se vuoi mantenere il tuo active-set
        return float(np.sum(r))

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


import numpy as np

class ReverseWaterFillingRateAllocator(RateAllocator):
    """
    Reverse water-filling (Gaussian, MSE) con vincolo per-modo r_i >= r_min.

    Modalità per estrarre i "modes":
      - mode="per_dim": sigma2_i = varianza della dimensione i
      - mode="pca":     sigma2_i = autovalori della covarianza (PCA)

    Vincolo aggiunto:
      - r_i >= r_min per OGNI modo considerato.
        Implementazione: baseline di r_min su tutti i modi -> D_cap_i = sigma2_i * 2^{-2 r_min}
        poi reverse water-filling sui bit residui su "sigma2' = D_cap".

    Outputs (self.optim) includono:
      - "Sigma2Modes": variances usate (per-mode)
      - "Sigma2AfterFloor": D_cap (varianze post-baseline)
      - "Theta": water level sui bit residui (applicato a D_cap)
      - "RatePerMode": rate finale per modo
      - "BitsPerMode": bit interi per modo (con floor a ceil(r_min) se abilitato)
      - "DistortionPerMode": D_i = min(D_cap_i, theta)
      - Se mode="pca": "V" e "Mean"
    """

    def __init__(self, R_tot, **kwargs):
        super().__init__(R_tot)
        self.rate_tol = kwargs.get("rate_tol", 1e-6)
        self.max_bisect_iter = kwargs.get("max_bisect_iter", 200)
        self.theta_min = kwargs.get("theta_min", 1e-18)
        self.theta_max_mult = kwargs.get("theta_max_mult", 1.0)
        self.lambda_floor = kwargs.get("lambda_floor", 1e-12)  # evita log2(0)

        # >>> vincolo per-modo
        self.r_min = float(kwargs.get("r_min", 1.0))

        # >>> opzionale: applica floor anche ai bit interi
        self.enforce_integer_floor = bool(kwargs.get("enforce_integer_floor", True))

        self.optim = {}

    # -----------------------------
    # Public API
    # -----------------------------
    def solve_allocation_problem(self, X, nz=None, mode="per_dim", center=True, rowvar=False):
        """
        Parameters
        ----------
        X : array-like
            Data matrix. Default shape (n_samples, n_features).
            If rowvar=True, expects (n_features, n_samples).
        nz : int or None
            Use only first nz modes (largest variances).
        mode : {"pca","per_dim"}
            How to define Gaussian modes from data.
        center : bool
            If True, subtract sample mean before estimating covariance/variances.
        rowvar : bool
            If True, rows are variables, columns are samples.

        Returns
        -------
        theta : float
            water level sui bit residui (applicato a D_cap)
        D : np.ndarray
            distortion per mode
        rate_per_mode : np.ndarray
            rate per mode (bits/sample)
        """

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (samples x features) or (features x samples).")

        # Put into (n_samples, n_features)
        Xs = X.T if rowvar else X

        n_samples, n_features = Xs.shape
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to estimate variances/covariance.")

        mean = Xs.mean(axis=0) if center else np.zeros(n_features, dtype=float)
        Xc = Xs - mean if center else Xs

        if mode == "per_dim":
            sigma2 = Xc.var(axis=0, ddof=1)
            V = None
        elif mode == "pca":
            C = (Xc.T @ Xc) / (n_samples - 1)
            evals, evecs = np.linalg.eigh(C)
            idx = np.argsort(evals)[::-1]
            sigma2 = evals[idx]
            V = evecs[:, idx]
        else:
            raise ValueError("mode must be 'pca' or 'per_dim'.")

        # floor per stabilità numerica
        sigma2 = np.maximum(sigma2, self.lambda_floor)

        # truncation
        if nz is not None:
            nz = int(nz)
            sigma2 = sigma2[:nz]
            if V is not None:
                V = V[:, :nz]
        else:
            nz = sigma2.size

        # --- Fattibilità del vincolo r_i >= r_min ---
        if self.r_min > 0 and self.R_tot < nz * self.r_min - self.rate_tol:
            raise ValueError(
                f"Infeasible: R_tot={self.R_tot} < nz*r_min={nz*self.r_min}."
            )

        # Baseline: impongo r_min su tutti i modi => D_cap
        D_cap = self._distortion_cap_from_rmin(sigma2, self.r_min)

        # Bit residui da allocare oltre al floor
        R_rem = float(self.R_tot - nz * self.r_min)

        # Solve theta sui bit residui usando reverse water-filling standard su sigma2' = D_cap
        theta = self._solve_theta_standard(D_cap, R_rem)

        # Distorsione finale (mai peggiore di D_cap)
        D = np.minimum(D_cap, theta)

        # Rate finale coerente: r_i = 0.5 log2(sigma2_i / D_i)
        rate_per_mode = self._rate_from_sigma2_D(sigma2, D)

        # Forza eventuali minuscole violazioni numeriche del floor
        if self.r_min > 0:
            rate_per_mode = np.maximum(rate_per_mode, self.r_min)

        bits = self._rates_to_integer_bits(rate_per_mode)
        if self.enforce_integer_floor and self.r_min > 0:
            bits_floor = int(np.ceil(self.r_min))
            bits = np.maximum(bits, bits_floor)

        self.optim = {
            "Mode": mode,
            "Mean": mean,
            "V": V,
            "Sigma2Modes": sigma2,
            "Sigma2AfterFloor": D_cap,
            "r_min": self.r_min,
            "R_rem": R_rem,
            "Theta": float(theta),
            "RatePerMode": rate_per_mode,
            "BitsPerMode": bits,
            "DistortionPerMode": D,
        }

        return float(theta), D, rate_per_mode

    # -----------------------------
    # Core: vincolo r_min
    # -----------------------------
    def _distortion_cap_from_rmin(self, sigma2, r_min):
        """
        D_cap_i = sigma2_i * 2^{-2 r_min}
        tale che 0.5 log2(sigma2_i / D_cap_i) = r_min
        """
        sigma2 = np.asarray(sigma2, dtype=float)
        r_min = float(r_min)

        if r_min <= 0.0:
            return sigma2.copy()

        # fattore 2^{-2 r_min}
        factor = 2.0 ** (-2.0 * r_min)
        D_cap = sigma2 * factor

        # evita zeri/negativi (stabilità numerica)
        tiny = np.finfo(float).tiny
        D_cap = np.maximum(D_cap, tiny)

        return D_cap

    # -----------------------------
    # Reverse water-filling standard (senza floor)
    # -----------------------------
    def _rate_from_sigma2_theta_standard(self, sigma2, theta):
        """
        Standard: r_i = 0.5 log2(sigma2_i/theta)+
        """
        sigma2 = np.asarray(sigma2, float)
        theta = float(theta)

        r = np.zeros_like(sigma2, dtype=float)
        # attivi se sigma2 > theta (theta>0)
        active = (theta > 0) & (sigma2 > theta)
        if np.any(active):
            r[active] = 0.5 * np.log2(sigma2[active] / theta)

        # non azzeriamo con rEps: qui è standard; serve per risolvere il residuo
        r[r < 0.0] = 0.0
        return r

    def _total_rate_standard(self, sigma2, theta):
        return float(np.sum(self._rate_from_sigma2_theta_standard(sigma2, theta)))

    def _solve_theta_standard(self, sigma2, R_target):
        """
        Trova theta tale che sum_i 0.5 log2(sigma2_i/theta)+ = R_target.

        Se R_target <= 0 -> theta = max(sigma2)*theta_max_mult (rate ~0).
        Se anche a theta_min non raggiungi R_target -> ritorna theta_min (saturazione).
        """
        sigma2 = np.asarray(sigma2, float)
        smax = float(np.max(sigma2)) if sigma2.size > 0 else 0.0

        # Nessun bit extra da allocare
        if R_target <= self.rate_tol or smax <= 0.0:
            return max(smax * self.theta_max_mult, self.theta_min)

        theta_hi = max(smax * self.theta_max_mult, self.theta_min)  # rate ~0
        R_hi = self._total_rate_standard(sigma2, theta_hi)

        theta_lo = self.theta_min  # rate massima (dato il floor numerico su theta)
        R_lo = self._total_rate_standard(sigma2, theta_lo)

        # Se anche al minimo theta non raggiungo R_target -> saturazione
        if R_lo < R_target - self.rate_tol:
            return theta_lo

        # Se già a theta_hi ho rate troppo alta (caso molto raro) -> ritorna theta_hi
        if R_hi > R_target + self.rate_tol:
            return theta_hi

        lo, hi = theta_lo, theta_hi
        for _ in range(self.max_bisect_iter):
            mid = 0.5 * (lo + hi)
            R_mid = self._total_rate_standard(sigma2, mid)

            if abs(R_mid - R_target) <= self.rate_tol:
                return mid

            if R_mid > R_target:
                # troppa rate -> aumenta theta
                lo = mid
            else:
                # poca rate -> diminuisci theta
                hi = mid

        return 0.5 * (lo + hi)

    # -----------------------------
    # Utilities: rate finale da (sigma2, D)
    # -----------------------------
    def _rate_from_sigma2_D(self, sigma2, D):
        sigma2 = np.asarray(sigma2, dtype=float)
        D = np.asarray(D, dtype=float)

        tiny = np.finfo(float).tiny
        D = np.maximum(D, tiny)
        sigma2 = np.maximum(sigma2, tiny)

        r = 0.5 * np.log2(sigma2 / D)
        # Non dovrebbe mai essere negativo, ma per sicurezza:
        r[r < 0.0] = 0.0
        return r

