import numpy as np
from mosek.fusion import Model, Expr, Domain, ObjectiveSense, Matrix
import logging
import time
from typing import Tuple

def _symmetrize_psd(C: np.ndarray, eps: float = 0.0) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    if eps > 0.0:
        C = C + eps * np.eye(C.shape[0])
    return C


def pca_factor_model_from_cov(C: np.ndarray, K: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a K-factor model approximation Σ ≈ B F Bᵀ + D from a raw covariance C using PCA.

    Returns
    -------
    B : (N, K) loadings (columns are factor loadings)
    F : (K, K) factor covariance (diagonal)
    D : (N,)   idiosyncratic variances (diagonal of residual)
    """
    C = _symmetrize_psd(C)
    N = C.shape[0]
    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(C)
    # Sort largest first
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Clip tiny/negative eigenvalues for stability
    vals = np.maximum(vals, 0.0)

    K_eff = int(min(K, N))
    if K_eff <= 0:
        # No factors: all diagonal
        B = np.zeros((N, 0), dtype=np.float64)
        F = np.zeros((0, 0), dtype=np.float64)
        D = np.clip(np.diag(C), 1e-12, None)
        return B, F, D

    # Top-K eigenpairs
    vals_K = vals[:K_eff]
    U_K = vecs[:, :K_eff]

    # Construct factor covariance and loadings
    # One common choice: Σ ≈ U_K diag(vals_K) U_Kᵀ + D, where D is the residual diagonal.
    # We set:
    #   F = diag(vals_K)
    #   B = U_K  (so B F Bᵀ = U_K diag(vals_K) U_Kᵀ)
    # Residual diagonal:
    recon_diag = np.sum((U_K**2) * vals_K[None, :], axis=1)
    D = np.clip(np.diag(C) - recon_diag, 1e-12, None)

    # Ensure numeric sanity
    F = np.diag(vals_K)
    B = U_K  # (N,K)

    return B, F, D

class SinglePeriodOptimizer:
    def __init__(self, risk_budget=0.10, gme_limit=2):
        self.risk_budget = risk_budget
        self.gme_limit = gme_limit
        self.K_factors = 20  # default number of factors for factor model

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


    # def _compute_cholesky_single(self, C):
    #     """
    #     Step 1: Compute Cholesky decomposition for a single covariance matrix.

    #     Parameters
    #     ----------
    #     C : (N,N) np.ndarray
    #         Covariance matrix (PSD)

    #     Returns
    #     -------
    #     L : (N,N) np.ndarray or None
    #         Cholesky decomposition, or None if decomposition fails
    #     """
    #     N = C.shape[0]
    #     try:
    #         L = np.linalg.cholesky(C)
    #         return L
    #     except np.linalg.LinAlgError:
    #         # try a tiny ridge to repair near-PSD matrices
    #         eps = 1e-10 * np.trace(C) / max(N, 1)
    #         try:
    #             L = np.linalg.cholesky(C + eps * np.eye(N))
    #             return L
    #         except np.linalg.LinAlgError:
    #             return None


    def _compute_cholesky_single(self, C):
        """
        Step 1: Compute Cholesky decomposition for a single covariance matrix.

        Parameters
        ----------
        C : (N,N) np.ndarray
            Covariance matrix (PSD)

        Returns
        -------
        L : (N,N) np.ndarray or None
            Cholesky factor (lower-triangular), or None if decomposition fails.
        """
        # Coerce to float64 to avoid integer dtypes and improve stability
        C = np.asarray(C, dtype=np.float64)
        N = C.shape[0]

        # Fast sanity checks
        if not np.all(np.isfinite(C)):
            return None
        if N == 0:
            return np.empty((0, 0), dtype=np.float64)

        # Symmetrize to kill tiny asymmetries
        C = 0.5 * (C + C.T)

        # Quick positive diagonal check (variance must be > 0)
        d = np.diag(C)
        if np.any(d <= 0):
            # Try to repair by shifting the diagonal
            shift = max(1e-12, 1e-10 * float(np.mean(d[d > 0])) if np.any(d > 0) else 1e-10)
            C = C + shift * np.eye(N)

        # 1) Try plain Cholesky
        try:
            return np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            pass

        # 2) Try escalating jitter (ridge) relative to scale
        # Use a scale based on average variance to be dimensionally consistent
        scale = float(np.mean(np.diag(C))) if np.all(np.isfinite(np.diag(C))) else 1.0
        base_eps = 1e-12 * max(scale, 1.0)
        eye = np.eye(N)
        for k in range(8):  # escalate up to ~1e-12 .. 1e-4 * scale
            eps = (10.0 ** k) * base_eps
            try:
                return np.linalg.cholesky(C + eps * eye)
            except np.linalg.LinAlgError:
                continue

        # 3) Final fallback: project to nearest SPD via eigenvalue clip
        try:
            vals, vecs = np.linalg.eigh(C)
            # Tolerance relative to spectrum and machine epsilon
            tol = max(1e-12 * scale, 1e-15)
            vals_clipped = np.maximum(vals, tol)
            C_spd = (vecs * vals_clipped) @ vecs.T
            # Symmetrize again to remove numeric skew
            C_spd = 0.5 * (C_spd + C_spd.T)
            return np.linalg.cholesky(C_spd)
        except Exception:
            # Give up
            return None

    def _formulate_sp_problem(self,
                            alpha,
                            L,
                            x0,
                            N,
                            B,                           # compulsory factor exposure matrix (K x N)
                            risk_lambda=1.0,             # default = 1
                            gme_limit=2.0,               # default = 1
                            position_bound=0.02,
                            trade_size_bound=0.01,
                            factor_neutral_tolerance=0.0):
        """
        Formulate the single-period optimization problem with factor neutrality.

        Parameters
        ----------
        alpha : array-like, shape (N,)
            Expected returns per asset.
        L : array-like or Fusion Matrix, shape (N,N) or (K_risk, N)
            Risk-loading such that ||L x||_2 gives portfolio volatility proxy.
        x0 : array-like, shape (N,)
            Starting portfolio weights.
        N : int
            Number of assets.
        B : array-like or Fusion Matrix, shape (K, N)
            Factor exposure matrix. Enforces factor neutrality B x ≈ 0.
        risk_lambda : float, default 1.0
            Risk-aversion multiplier on r (epigraph of ||Lx||_2).
        gme_limit : float, default 1.0
            Gross market exposure cap: sum_i |x_i| ≤ gme_limit.
        position_bound : float, default 0.02
            Symmetric per-asset position bound: |x_i| ≤ position_bound.
        trade_size_bound : float, default 0.01
            Symmetric per-asset trade bound: |x_i - x0_i| ≤ trade_size_bound.
        factor_neutral_tolerance : float, default 0.0
            If 0, exact neutrality per factor: (B x)_k = 0.
            If > 0, banded neutrality: -tol ≤ (B x)_k ≤ tol.

        Returns
        -------
        M : mosek.fusion.Model
            The formulated optimization model (not yet solved).
        x : mosek.fusion.Variable
            The decision variable (weights).
        """
        logger = logging.getLogger(__name__)

        L = np.asarray(L, dtype=np.float64)
        t_start = time.perf_counter()

        # Basic input checks
        if hasattr(alpha, "shape"):
            assert alpha.shape[0] == N, f"alpha length {len(alpha)} != N {N}"
        if hasattr(x0, "shape"):
            assert x0.shape[0] == N, f"x0 length {len(x0)} != N {N}"

        # Convert L to Fusion Matrix if passed as numpy
        t0 = time.perf_counter()
        if hasattr(L, "shape"):
            assert L.shape[1] == N, f"L must have N={N} columns, got {L.shape}"
            L_mat = Matrix.dense(L.tolist())
        else:
            L_mat = L  # assume Fusion Matrix with N columns
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Convert L to Matrix: {(t1-t0)*1000:.2f}ms")

        # Convert B to Fusion Matrix if passed as numpy
        t0 = time.perf_counter()
        if hasattr(B, "shape"):
            K, BN = B.shape
            assert BN == N, f"B must be K×N with N={N}, got {B.shape}"
            B_mat = Matrix.dense(B.tolist())
        else:
            B_mat = B
            # If B is Fusion Matrix, we won't know K here; we'll handle tolerance accordingly.
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Convert B to Matrix: {(t1-t0)*1000:.2f}ms")

        # Create model
        t0 = time.perf_counter()
        M = Model("LongShortPortfolio_RiskPenalty")
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Create Model: {(t1-t0)*1000:.2f}ms")

        # Decision variables
        t0 = time.perf_counter()
        x = M.variable("x", N, Domain.inRange(-position_bound, position_bound))  # portfolio weights
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Variable x (N={N}): {(t1-t0)*1000:.2f}ms")

        t0 = time.perf_counter()
        r = M.variable("r", 1, Domain.greaterThan(0.0))                          # risk epigraph
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Variable r: {(t1-t0)*1000:.2f}ms")

        t0 = time.perf_counter()
        t = M.variable("t", N, Domain.greaterThan(0.0))                          # |x| helper for GME
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Variable t (N={N}): {(t1-t0)*1000:.2f}ms")

        # |x| via t >= ±x
        t0 = time.perf_counter()
        M.constraint("abs_pos", Expr.sub(t, x),           Domain.greaterThan(0.0))
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Constraint abs_pos: {(t1-t0)*1000:.2f}ms")

        t0 = time.perf_counter()
        M.constraint("abs_neg", Expr.sub(t, Expr.neg(x)), Domain.greaterThan(0.0))
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Constraint abs_neg: {(t1-t0)*1000:.2f}ms")

        # Market neutrality
        t0 = time.perf_counter()
        M.constraint("market_neutral", Expr.sum(x), Domain.equalsTo(0.0))
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Constraint market_neutral: {(t1-t0)*1000:.2f}ms")

        # Gross market exposure
        t0 = time.perf_counter()
        M.constraint("gme_constraint", Expr.sum(t), Domain.lessThan(gme_limit))
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Constraint gme_constraint: {(t1-t0)*1000:.2f}ms")

        # Risk epigraph: r >= ||L x||_2   (SOC)
        t0 = time.perf_counter()
        M.constraint("risk_soc", Expr.vstack(r, Expr.mul(L_mat, x)), Domain.inQCone())
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Constraint risk_soc: {(t1-t0)*1000:.2f}ms")

        # Per-ticker trade bounds: -b <= x - x0 <= b
        t0 = time.perf_counter()
        x0_list = x0.tolist() if hasattr(x0, "tolist") else list(x0)
        M.constraint("delta_x", Expr.sub(x, x0_list), Domain.inRange(-trade_size_bound, trade_size_bound))
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Constraint delta_x: {(t1-t0)*1000:.2f}ms")

        # Factor neutrality: B x = 0 (exact) or within tolerance
        t0 = time.perf_counter()
        Bx = Expr.mul(B_mat, x)
        if factor_neutral_tolerance <= 0.0:
            M.constraint("factor_neutral_exact", Bx, Domain.equalsTo(0.0))
        else:
            tol = factor_neutral_tolerance
            # If we know K (numpy path), use vector bounds; else use scalar band (broadcast)
            if 'K' in locals():
                lower = [-tol] * K
                upper = [ tol] * K
                M.constraint("factor_neutral_band", Bx, Domain.inRange(lower, upper))
            else:
                M.constraint("factor_neutral_band", Bx, Domain.inRange(-tol, tol))
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Constraint factor_neutral: {(t1-t0)*1000:.2f}ms")

        # Objective: maximize alpha'x - risk_lambda * r
        t0 = time.perf_counter()
        pnl_term  = Expr.dot(alpha, x)
        risk_term = Expr.mul(risk_lambda, r)
        M.objective("obj", ObjectiveSense.Maximize, Expr.sub(pnl_term, risk_term))
        t1 = time.perf_counter()
        logger.info(f"[Formulation] Objective: {(t1-t0)*1000:.2f}ms")

        t_end = time.perf_counter()
        logger.info(f"[Formulation] TOTAL formulation time: {(t_end-t_start)*1000:.2f}ms")

        return M, x

    def _formulate_sp_problem_factor(
        self,
        alpha: np.ndarray,
        B: np.ndarray,          # (N, K)
        F: np.ndarray,          # (K, K) diagonal
        D: np.ndarray,          # (N,)   diagonal
        x0: np.ndarray,
        N: int,
        risk_lambda: float,
        gme_limit: float,
        position_bound: float,
        trade_size_bound: float,
        B_neutral: np.ndarray | None = None,
        factor_neutral_tolerance: float = 0.0,
    ):
        """
        Factor-model formulation:
          maximize   alpha' x - risk_lambda * r
          s.t.       sum(x) = 0
                     y = sqrt(F) Bᵀ x
                     z = sqrt(D) ⊙ x
                     r ≥ ||[y; z]||₂
                     sum |x| ≤ gme_limit
                     |x_i| ≤ position_bound
                     |x_i - x0_i| ≤ trade_size_bound
                     B_neutral x ≈ 0 (optional)
        """
        logger = logging.getLogger(__name__)
        t_start = time.perf_counter()

        K = B.shape[1] if B.size > 0 else 0
        assert F.shape == (K, K)
        assert D.shape == (N,)
        if K > 0:
            # sqrt(F) from diagonal
            F_diag = np.diag(F).copy()
            F_sqrt = np.sqrt(np.maximum(F_diag, 0.0))
            # Precompute M = diag(sqrt(F)) @ Bᵀ  -> shape (K, N)
            M_fac = (F_sqrt[:, None] * B.T)  # (K,N)
            M_fac_mat = Matrix.dense(M_fac)  # Fusion dense matrix (small K×N)
        else:
            M_fac_mat = None

        s = np.sqrt(np.maximum(D, 0.0))  # length-N, for z = s ⊙ x

        # Create model
        M = Model("LongShortPortfolio_Factor")
        x = M.variable("x", N, Domain.inRange(-position_bound, position_bound))
        r = M.variable("r", 1, Domain.greaterThan(0.0))
        t = M.variable("t", N, Domain.greaterThan(0.0))

        # |x| via t >= ±x
        M.constraint("abs_pos", Expr.sub(t, x),           Domain.greaterThan(0.0))
        M.constraint("abs_neg", Expr.sub(t, Expr.neg(x)), Domain.greaterThan(0.0))
        M.constraint("gme_constraint", Expr.sum(t), Domain.lessThan(gme_limit))

        # Market neutrality
        M.constraint("market_neutral", Expr.sum(x), Domain.equalsTo(0.0))

        # Trade bounds: -b <= x - x0 <= b
        if trade_size_bound is not None and np.isfinite(trade_size_bound):
            M.constraint("delta_x", Expr.sub(x, x0.tolist()), Domain.inRange(-trade_size_bound, trade_size_bound))

        # Optional factor neutrality constraints (separate from risk factors)
        if B_neutral is not None and B_neutral.size > 0:
            Bn = np.asarray(B_neutral, dtype=np.float64)
            Kneu, Nneu = Bn.shape
            if Nneu != N:
                raise ValueError(f"B_neutral must have N={N} columns, got {Nneu}")
            Bn_mat = Matrix.dense(Bn)
            Bx = Expr.mul(Bn_mat, x)
            if factor_neutral_tolerance <= 0.0:
                M.constraint("factor_neutral_exact", Bx, Domain.equalsTo(0.0))
            else:
                lower = [-factor_neutral_tolerance] * Kneu
                upper = [ factor_neutral_tolerance] * Kneu
                M.constraint("factor_neutral_band", Bx, Domain.inRange(lower, upper))

        # Build risk SOC via factor model
        # y = M_fac x  (K)   where M_fac = sqrt(F) Bᵀ
        # z = s ⊙ x    (N)
        if K > 0:
            y = Expr.mul(M_fac_mat, x)  # dimension K
            # Stack [r; y; s ⊙ x] into one SOC
            soc_arg = Expr.vstack(r, y, Expr.mulElm(s.tolist(), x))
        else:
            soc_arg = Expr.vstack(r, Expr.mulElm(s.tolist(), x))
        M.constraint("risk_soc", soc_arg, Domain.inQCone())
        M.constraint("risk_cap", r, Domain.lessThan(self.risk_budget))  # or pass risk_budget as arg

        # Objective
        pnl_term  = Expr.dot(alpha, x)
        risk_term = Expr.mul(risk_lambda, r)
        M.objective("obj", ObjectiveSense.Maximize, Expr.sub(pnl_term, risk_term))

        t_end = time.perf_counter()
        logger = logging.getLogger(__name__)
        logger.info(f"[Formulation-Factor] TOTAL formulation time: {(t_end - t_start)*1000:.2f} ms")
        return M, x

    def _solve_sp_model(self, M, x):
        """
        Step 3: Solve the single-period optimization model.

        Returns
        -------
        result : (N,) np.ndarray
            Optimal weights
        solve_time : float
            Time taken to solve (in seconds)
        """
        solve_start = time.perf_counter()
        M.solve()
        solve_time = time.perf_counter() - solve_start

        result = np.array(x.level())
        M.dispose()

        return result, solve_time

    # def solve_long_short_portfolio_with_risk_penalty(
    #     self,
    #     alpha: np.ndarray,
    #     C: np.ndarray,
    #     x0: np.ndarray = None,        # starting portfolio (default: zeros)
    #     B: np.ndarray = None,         # factor exposure matrix (default: none)
    #     risk_lambda: float = 1.0,     # multiplier on risk penalty
    #     gme_limit: float = 1.0,       # sum |x| <= gme_limit  (in weight units)
    #     position_bound: float = 0.02, # |x_i| <= position_bound
    #     trade_size_bound: float = 0.01, # |x_i - x0_i| <= trade_size_bound (default: inf)
    #     factor_neutral_tolerance: float = 0.0  # factor neutrality tolerance
    # ) -> np.ndarray:
    #     """
    #     Long-short, market-neutral optimizer with risk penalty in the objective and optional factor constraints.

    #     maximize   alpha' x  -  risk_lambda * r
    #     s.t.       sum(x) = 0
    #                r >= || L x ||_2
    #                sum |x| <= gme_limit
    #                -position_bound <= x_i <= position_bound
    #                |x_i - x0_i| <= trade_size_bound
    #                B x ≈ 0  (factor neutrality)

    #     Parameters
    #     ----------
    #     alpha : (N,) signal vector
    #     C     : (N,N) covariance (PSD)
    #     x0    : (N,) starting portfolio weights (default: zeros)
    #     B     : (K,N) factor exposure matrix (default: none, no factor constraints)
    #     risk_lambda : scalar >= 0, penalty multiplier on risk magnitude r
    #     gme_limit   : scalar > 0, gross market exposure limit in weight units
    #     position_bound : scalar > 0, box bounds on each x_i
    #     trade_size_bound : scalar > 0 or None, trade size bound (default: inf = no bound)
    #     factor_neutral_tolerance : scalar >= 0, tolerance for factor neutrality

    #     Returns
    #     -------
    #     x_opt : (N,) optimized weights
    #     """
    #     logger = logging.getLogger(__name__)

    #     alpha = np.asarray(alpha, dtype=np.float64)
    #     N = alpha.size
    #     if C.shape != (N, N):
    #         raise ValueError(f"Covariance matrix shape {C.shape} does not match alpha size {N}.")
    #     if alpha.ndim != 1:
    #         raise ValueError(f"alpha must be 1D, got {alpha.shape}.")

    #     # Default x0 to zeros
    #     if x0 is None:
    #         x0 = np.zeros(N)
    #     else:
    #         x0 = np.asarray(x0, dtype=np.float64)

    #     # Default B to empty matrix (no factor constraints)
    #     if B is None:
    #         B = np.zeros((0, N))
    #     else:
    #         B = np.asarray(B, dtype=np.float64)


    #     # Step 1: Compute Cholesky decomposition
    #     L = self._compute_cholesky_single(C)
    #     if L is None:
    #         return np.zeros(N)

    #     T = time.perf_counter()
    #     # Step 2: Formulate the optimization problem
    #     M, x = self._formulate_sp_problem(
    #         alpha=alpha,
    #         L=L,
    #         x0=x0,
    #         N=N,
    #         B=B,
    #         risk_lambda=risk_lambda,
    #         gme_limit=gme_limit,
    #         position_bound=position_bound,
    #         trade_size_bound=trade_size_bound,
    #         factor_neutral_tolerance=factor_neutral_tolerance
    #     )

    #     # Step 3: Solve the model
    #     result, solve_time = self._solve_sp_model(M, x)
    #     endT = time.perf_counter()

    #     logger.info(f"[SP Optimizer] N={N} | Solve={solve_time:.4f}s")

    #     logger.info(f"Total time is {endT - T}s")

    #     return result

    def solve_long_short_portfolio_with_risk_penalty(
            self,
            alpha: np.ndarray,
            C: np.ndarray,
            x0: np.ndarray = None,        # starting portfolio (default: zeros)
            B: np.ndarray = None,         # factor exposure matrix for neutrality (optional, separate from risk factors)
            risk_lambda: float = 1.0,     # multiplier on risk penalty
            gme_limit: float = 2.0,       # sum |x| <= gme_limit  (in weight units)
            position_bound: float = 0.02, # |x_i| <= position_bound
            trade_size_bound: float = 0.01, # |x_i - x0_i| <= trade_size_bound
            factor_neutral_tolerance: float = 0.0  # neutrality tolerance
        ) -> np.ndarray:
            """
            Long-short, market-neutral optimizer with factor-model risk:
            Σ ≈ B F Bᵀ + D via PCA with K=self.K_factors (default 20)
            """
            logger = logging.getLogger(__name__)

            alpha = np.asarray(alpha, dtype=np.float64)
            N = alpha.size
            C = np.asarray(C, dtype=np.float64)
            if C.shape != (N, N):
                raise ValueError(f"Covariance matrix shape {C.shape} does not match alpha size {N}.")
            if alpha.ndim != 1:
                raise ValueError(f"alpha must be 1D, got {alpha.shape}.")

            if x0 is None:
                x0 = np.zeros(N, dtype=np.float64)
            else:
                x0 = np.asarray(x0, dtype=np.float64)

            # Build factor model from raw covariance
            B_risk, F_risk, D_risk = pca_factor_model_from_cov(C, K=self.K_factors)

            # Formulate factor-model problem
            t_form_start = time.perf_counter()
            M, x = self._formulate_sp_problem_factor(
                alpha=alpha,
                B=B_risk,
                F=F_risk,
                D=D_risk,
                x0=x0,
                N=N,
                risk_lambda=risk_lambda,
                gme_limit=gme_limit,
                position_bound=position_bound,
                trade_size_bound=trade_size_bound,
                B_neutral=B,  # optional neutrality matrix provided by caller
                factor_neutral_tolerance=factor_neutral_tolerance
            )
            t_form_end = time.perf_counter()

            # Solve
            result, solve_time = self._solve_sp_model(M, x)
            formulation_time = t_form_end - t_form_start
            logger.info(f"[SP Optimizer] N={N} | K={self.K_factors} | Formulation={formulation_time:.4f}s | Solve={solve_time:.4f}s")

            # Store timing info for external access
            self.last_formulation_time = formulation_time
            self.last_solve_time = solve_time

            return result

    @staticmethod
    def nearest_psd(A, epsilon=1e-12):
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
