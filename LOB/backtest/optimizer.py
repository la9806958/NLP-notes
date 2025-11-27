import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mosek.fusion import Model, Expr, Domain, ObjectiveSense

class Optimizer:
    def __init__(self, risk_budget=0.01, gme_limit=2):
        self.risk_budget = risk_budget
        self.gme_limit = gme_limit
        self.lookback_cov = 100 # days of cached alphas used to optimize

    def optimise(self,all_dates,i,alpha_cache):

        hist_dates = all_dates[i - self.lookback_cov: i]
        
        alpha_hist = alpha_cache.loc[hist_dates].dropna(axis=1, how='any')
        alpha_hist = alpha_hist.apply(pd.to_numeric, errors='coerce')

        C_df = alpha_hist.cov()
        C_df = C_df.loc[(C_df != 0).any(axis=1), (C_df != 0).any(axis=0)]
        common_syms = alpha_hist.columns.intersection(C_df.columns)
        alpha_hist = alpha_hist[common_syms]
        C_df = C_df.loc[common_syms, common_syms]
        C_df *= 252

        # Zero out off-diagonal covariance elements here
        C_df = pd.DataFrame(np.diag(np.diag(C_df)), index=C_df.index, columns=C_df.columns)

        alpha_today_series = alpha_cache.loc[d, common_syms].dropna()
        common_syms = C_df.columns.intersection(alpha_today_series.index)
        C_df = C_df.loc[common_syms, common_syms]
        alpha_today_series = alpha_today_series.loc[common_syms]

        if len(common_syms) == 0:
            print(f"Skipping {d} - no valid symbols after cleanup.")
            return None
        
        C = Optimizer.nearest_psd(A=C_df)
        C_df = pd.DataFrame(C, index=C_df.index, columns=C_df.columns)
        
        alpha_for_opt = alpha_today_series.values
        C_for_opt = C_df.values

        try:
            wts = Optimizer.solve_long_short_portfolio(alpha_for_opt, C_for_opt)
        except np.linalg.LinAlgError:
            print(f"Skipping {d} - covariance matrix inversion failed.")
            return None

        rec = {"date": d}
        rec.update(dict(zip(common_syms, wts)))
        return rec


    def solve_long_only_portfolio(self,
                                  alpha: np.ndarray,    # expected returns r  (N,)
                                  C:     np.ndarray     # full covariance Σ  (N×N)
                                 ) -> np.ndarray:
        """
        Compute long-only weights w such that:
          • direction is arg-max Sharpe under a *diagonal* Σ   (w_i ∝ r_i / σ_i²)
          • ex-ante volatility under the *full* Σ equals TARGET_RISK.

        Returns
        -------
        w : (N,) ndarray
            Levered long-only weight vector.  `w.sum()` is the leverage required
            to hit the risk target; it need not equal 1.
        """
        # --- defensive copies / shape checks --------------------------------
        alpha = np.asarray(alpha, dtype=float).ravel()
        C     = np.asarray(C,     dtype=float)
        if C.shape[0] != C.shape[1] or C.shape[0] != alpha.size:
            raise ValueError("Dimension mismatch between alpha and covariance.")

        # --------------------------------------------------------------------
        # 1) Diagonal max-Sharpe   w_i ∝ r_i / σ_i²  (guaranteed non-negative)
        # --------------------------------------------------------------------
        variances = np.diag(C)
        if np.any(variances <= 0):
            raise ValueError("Covariance matrix must have strictly positive variances.")

        w = alpha / variances                    # r_i / σ_i²
        w = np.clip(w, 0.0, None)                # just in case alpha has negatives
        if w.sum() == 0:
            return w
        w /= w.sum()                             # impose 1ᵀw = 1  (pure direction)

        # --------------------------------------------------------------------
        # 2) Rescale so that √(wᵀ Σ_full w) = TARGET_RISK
        # --------------------------------------------------------------------
        #portfolio_vol = np.sqrt(w @ C @ w)       # use *full* Σ here
        #if portfolio_vol == 0:
        #    raise ValueError("Degenerate portfolio variance (Σ may be singular).")

        #leverage = self.risk_budget / portfolio_vol
        #w *= leverage                            # final levered weights

        return w

        
    def solve_long_short_portfolio(self, alpha: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Solve a long-short, market-neutral portfolio optimization:
        maximize alpha' x
        subject to sum(x)=0, || L*x ||_2 <= risk_budget, sum(|x|) <= gme_limit

        :param alpha: (N,) array of signals
        :param C: (N, N) covariance matrix
        :return: (N,) array of optimized weights
        """
        alpha = np.asarray(alpha, dtype=np.float64)

        N = alpha.size
        if C.shape != (N, N):
            raise ValueError(f"Covariance matrix shape {C.shape} does not match alpha size {N}.")

        if alpha.ndim != 1:
            raise ValueError(f"alpha must be a 1D array, but got shape {alpha.shape}.")

        # Check covariance matrix positive semi-definiteness via Cholesky
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # Covariance matrix not PSD, return zero weights
            return np.zeros(N)

        with Model("LongShortPortfolio") as M:
            x = M.variable("x", N, Domain.inRange(-1, 1))

            # Market neutrality constraint
            M.constraint("market_neutral", Expr.sum(x), Domain.equalsTo(0.0))

            # Objective function
            M.objective("obj", ObjectiveSense.Maximize, Expr.dot(alpha, x))

            # Risk constraint
            M.constraint("risk", Expr.vstack(Expr.constTerm(self.risk_budget), Expr.mul(L, x)), Domain.inQCone())

            # Gross Market Exposure constraint
            t = M.variable("t", N, Domain.greaterThan(0.0))
            M.constraint("abs_pos", Expr.sub(t, x), Domain.greaterThan(0.0))
            M.constraint("abs_neg", Expr.sub(t, Expr.neg(x)), Domain.greaterThan(0.0))
            M.constraint("gme_constraint", Expr.sum(t), Domain.lessThan(self.gme_limit))

            M.solve()

            return np.array(x.level())

    def nearest_psd(self,A, epsilon=1e-12):
        """
        Project a general square matrix A onto the PSD cone by:
          1) Making A symmetric,
          2) Eigen-decomposing,
          3) Clipping negative eigenvalues to 'epsilon',
          4) Reconstructing.
        """
        A_sym = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(A_sym)
        eigvals_clipped = np.clip(eigvals, a_min=epsilon, a_max=None)
        A_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        A_psd = 0.5 * (A_psd + A_psd.T)
        A_psd += np.eye(A_psd.shape[0]) * epsilon
        return A_psd

    def smooth_alpha_cache(self, alpha_cache, halflife):
        smoothed_df = alpha_cache.copy()
        alpha = 2**(-1/halflife)
        for col in smoothed_df.columns:
            smoothed_df[col] = smoothed_df[col].ewm(alpha=alpha, adjust=False).mean()

        return smoothed_df

    