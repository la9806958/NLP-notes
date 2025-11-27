import numpy as np
import pandas as pd
from mosek.fusion import Model, Expr, Domain, ObjectiveSense, Matrix
import logging
import time


class MultiPeriodOptimizer:
    def __init__(self, risk_budget=0.01, gme_limit=2):
        self.risk_budget = risk_budget
        self.gme_limit = gme_limit

        # Cache for reusable multi-period model
        self._mp_model_cache = None
        self._mp_model_x = None
        self._mp_model_L_params = None
        self._mp_model_alpha_params = None
        self._mp_model_config = None  # (N, T, risk_lambda, capital, gme_limit, box)
        self._mp_last_solution = None  # for warm start



    def _compute_cholesky_decompositions(self, covs):
        """
        Step 1: Compute Cholesky decompositions for all covariance matrices.

        Parameters
        ----------
        covs : list of (N,N) np.ndarray
            Covariance matrices per period (PSD)

        Returns
        -------
        Ls : list of (N,N) np.ndarray
            Cholesky decompositions per period
        """
        N = covs[0].shape[0]
        Ls = []
        for t, C in enumerate(covs):
            try:
                Lt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eps = 1e-10 * max(np.trace(C), 1.0) / max(N, 1)
                Lt = np.linalg.cholesky(C + eps * np.eye(N))
            Ls.append(Lt)
        return Ls

    def _formulate_mp_problem(self, alphas, Ls, N, T, risk_lambda, capital, gme_limit, box):
        """
        Step 2: Formulate the multi-period optimization problem.

        Returns
        -------
        M : mosek.fusion.Model
            The formulated optimization model (not yet solved)
        """
        logger = logging.getLogger(__name__)
        formulate_start = time.perf_counter()

        M = Model("MP_LongShort_RiskPenalty")

        # Decision variables
        x = M.variable("x", [N, T], Domain.inRange(-box, box))
        r = M.variable("r", T, Domain.greaterThan(0.0))
        u = M.variable("u", [N, T], Domain.greaterThan(0.0))  # |x| helper for GME

        for t in range(T):
            # Get column t for x and u using slice
            x_t = x.slice([0, t], [N, t+1])
            u_t = u.slice([0, t], [N, t+1])

            # Market neutrality: 1' x_t = 0
            M.constraint(f"neutral_{t}", Expr.sum(x_t), Domain.equalsTo(0.0))

            # |x| via u >= ±x
            M.constraint(f"abs_pos_{t}", Expr.sub(u_t, x_t), Domain.greaterThan(0.0))
            M.constraint(f"abs_neg_{t}", Expr.sub(u_t, Expr.neg(x_t)), Domain.greaterThan(0.0))

            # Gross exposure
            M.constraint(f"gme_{t}", Expr.sum(u_t), Domain.lessThan(gme_limit))

            # Risk epigraph: r_t >= ||L_t x_t||_2
            M.constraint(f"risk_{t}",
                        Expr.flatten(Expr.vstack(r.index(t),
                                                 Expr.mul(Ls[t], x_t))),   # x_t is (N,1)
                        Domain.inQCone())

        # Objective: sum_t [ capital*(alpha_t'x_t - risk_lambda*r_t) ]
        terms = []
        for t in range(T):
            x_t = x.slice([0, t], [N, t+1])
            pnl_t  = Expr.mul(capital, Expr.dot(alphas[t], x_t))
            risk_t = Expr.mul(capital * risk_lambda, r.index(t))
            terms.append(Expr.sub(pnl_t, risk_t))

        M.objective("objective", ObjectiveSense.Maximize, Expr.add(terms))

        formulate_time = time.perf_counter() - formulate_start
        logger.info(f"[MP Formulate] N={N}, T={T} | Formulate={formulate_time:.4f}s")

        return M, x

    def _build_mp_model_reusable(self, N, T, risk_lambda, capital, gme_limit, box):
        """
        Build a reusable multi-period optimization model with parameter variables.
        Model structure is built once and can be solved multiple times with updated L and alphas.

        Parameters
        ----------
        N : int
            Number of assets
        T : int
            Number of periods
        risk_lambda : float
            Risk penalty multiplier
        capital : float
            Capital scaling
        gme_limit : float
            Gross market exposure limit
        box : float
            Box constraint for positions

        Returns
        -------
        M : mosek.fusion.Model
            The model with parameter variables
        x : mosek.fusion.Variable
            Decision variable for positions
        L_params : list of mosek.fusion.Parameter
            Parameter objects for L matrix (one N×N matrix per period)
        alpha_params : list of mosek.fusion.Parameter
            Parameter objects for alphas (one N-vector per period)
        """
        logger = logging.getLogger(__name__)
        formulate_start = time.perf_counter()

        M = Model("MP_LongShort_RiskPenalty_Reusable")

        # Decision variables
        x = M.variable("x", [N, T], Domain.inRange(-box, box))
        r = M.variable("r", T, Domain.greaterThan(0.0))
        u = M.variable("u", [N, T], Domain.greaterThan(0.0))  # |x| helper for GME

        # Parameters (data that will be updated at each solve)
        alpha_params = [M.parameter(f"alpha_{t}", N) for t in range(T)]
        L_params = [M.parameter(f"L_{t}", [N, N]) for t in range(T)]

        # Constraints (structure only, no data dependencies)
        for t in range(T):
            # Get column t for x and u using slice
            x_t = x.slice([0, t], [N, t+1])
            u_t = u.slice([0, t], [N, t+1])

            # Market neutrality: 1' x_t = 0
            M.constraint(f"neutral_{t}", Expr.sum(x_t), Domain.equalsTo(0.0))

            # |x| via u >= ±x
            M.constraint(f"abs_pos_{t}", Expr.sub(u_t, x_t), Domain.greaterThan(0.0))
            M.constraint(f"abs_neg_{t}", Expr.sub(u_t, Expr.neg(x_t)), Domain.greaterThan(0.0))

            # Gross exposure
            M.constraint(f"gme_{t}", Expr.sum(u_t), Domain.lessThan(gme_limit))

            # Risk epigraph: r_t >= ||L_t x_t||_2  (L_t is parameter per period)
            M.constraint(f"risk_{t}",
                        Expr.flatten(Expr.vstack(r.index(t),
                                                 Expr.mul(L_params[t], x_t))),
                        Domain.inQCone())

        # Objective: sum_t [ capital*(alpha_t'x_t - risk_lambda*r_t) ]
        # Use parameters for alpha - flatten x_t slice to vector for dot product
        terms = []
        for t in range(T):
            x_t = x.slice([0, t], [N, t+1])           # shape [N, 1]
            x_t_vec = Expr.flatten(x_t)               # shape [N]
            pnl_t  = Expr.mul(capital, Expr.dot(alpha_params[t], x_t_vec))
            risk_t = Expr.mul(capital * risk_lambda, r.index(t))
            terms.append(Expr.sub(pnl_t, risk_t))

        M.objective("objective", ObjectiveSense.Maximize, Expr.add(terms))

        formulate_time = time.perf_counter() - formulate_start
        logger.info(f"[MP Build Reusable] N={N}, T={T} | Build={formulate_time:.4f}s")

        return M, x, L_params, alpha_params

    def _solve_mp_model_reusable(self, M, x, L_params, alpha_params, Ls, alphas, N, T, x_warmstart=None):
        """
        Solve a reusable model by updating L and alpha parameter variables and optionally using warm start.

        Parameters
        ----------
        M : mosek.fusion.Model
            The pre-built model
        x : mosek.fusion.Variable
            Decision variable
        L_params : list of mosek.fusion.Parameter
            Parameter objects for L matrices
        alpha_params : list of mosek.fusion.Parameter
            Parameter objects for alphas
        Ls : list of (N,N) np.ndarray
            Cholesky decompositions for this solve
        alphas : list of (N,) np.ndarray
            Alpha values for this solve
        N : int
            Number of assets
        T : int
            Number of periods
        x_warmstart : (N, T) np.ndarray, optional
            Warm start solution from previous solve

        Returns
        -------
        result : (N, T) np.ndarray
            Optimal weights per period
        solve_time : float
            Time taken to solve
        """
        logger = logging.getLogger(__name__)

        # Update parameters with new data (Parameters use setValue)
        for t in range(T):
            L_params[t].setValue(Ls[t])
            alpha_params[t].setValue(alphas[t])

        # Set warm start if provided
        if x_warmstart is not None:
            x.setLevel(x_warmstart.ravel())
            logger.info(f"[MP Solve Reusable] Using warm start")

        # Solve
        solve_start = time.perf_counter()
        M.solve()
        solve_time = time.perf_counter() - solve_start

        result = np.array(x.level()).reshape(N, T)

        # Parameters don't need to be unfixed - they'll be updated next solve
        return result, solve_time

    def _solve_mp_model(self, M, x, N, T, x_warmstart=None):
        """
        Step 3: Solve the optimization model.

        Parameters
        ----------
        M : mosek.fusion.Model
            The optimization model
        x : mosek.fusion.Variable
            Decision variable
        N : int
            Number of assets
        T : int
            Number of periods
        x_warmstart : (N, T) np.ndarray, optional
            Warm start solution from previous solve

        Returns
        -------
        result : (N, T) np.ndarray
            Optimal weights per period
        solve_time : float
            Time taken to solve (in seconds)
        """
        logger = logging.getLogger(__name__)

        # Set warm start if provided
        if x_warmstart is not None:
            x.setLevel(x_warmstart.ravel())
            logger.info(f"[MP Solve] Using warm start from previous solution")

        solve_start = time.perf_counter()
        M.solve()
        solve_time = time.perf_counter() - solve_start

        result = np.array(x.level()).reshape(N, T)
        M.dispose()

        return result, solve_time

    def formulate_mp_no_loops_same_L(self, alphas, L, N, T, risk_lambda, capital, gme_limit, box):
        """
        Vectorized MPO (no loop over t), assuming the same risk-loading L for all periods.
        alphas: (N, T) numpy array
        L:      (K, N) numpy array  (e.g., Cholesky or any risk-loading matrix)
        """
        import time, logging
        from mosek.fusion import Model, Matrix, Domain, Expr, ObjectiveSense
        import numpy as np

        logger = logging.getLogger(__name__)
        formulate_start = time.perf_counter()

        # Dimensions
        K = L.shape[0]
        if L.shape[1] != N:
            raise ValueError(f"L is shape {L.shape}, but expected (K, N) with N={N}.")
        if alphas.shape != (N, T):
            raise ValueError(f"alphas is shape {alphas.shape}, but expected (N, T)=({N}, {T}).")

        Alpha = Matrix.dense(alphas)

        M = Model("MP_LongShort_RiskPenalty_vec")

        # Decision vars
        X = M.variable("x", [N, T], Domain.inRange(-box, box))          # positions per period
        U = M.variable("u", [N, T], Domain.greaterThan(0.0))            # |x| helper
        R = M.variable("r",  T,       Domain.greaterThan(0.0))          # SOC epigraph radii per period

        # |x| via U >= ±X (elementwise)
        M.constraint("abs_pos", Expr.sub(U, X),            Domain.greaterThan(0.0))
        M.constraint("abs_neg", Expr.sub(U, Expr.neg(X)),  Domain.greaterThan(0.0))

        # Market neutrality: sum_i x_{i,t} = 0  for all t
        # sum over rows -> shape (1, T); flatten to vector length T for the domain
        M.constraint(
            "neutral",
            Expr.flatten(Expr.sum(X, 0)),
            Domain.equalsTo(np.zeros(T))
        )

        # Gross exposure per period: sum_i |x_{i,t}| <= gme_limit
        M.constraint(
            "gme",
            Expr.flatten(Expr.sum(U, 0)),
            Domain.lessThan(np.full(T, gme_limit))
        )

        # Risk SOCs for all t at once:
        # Y = L @ X  -> (K, T); V = [R; Y] -> (K+1, T).
        Y = Expr.mul(Matrix.dense(L), X)      # (K, T)
        R_reshaped = Expr.reshape(R, [1, T])  # reshape (T,) to (1, T)
        V = Expr.vstack(R_reshaped, Y)        # (K+1, T)
        # One SOC per column (each period): ||(L x_t)||_2 <= r_t
        M.constraint("risk_soc", V, Domain.inQCone().axis(0))

        # Objective: maximize capital * [ sum_{i,t} alpha_{i,t} x_{i,t} - risk_lambda * sum_t r_t ]
        pnl  = Expr.mul(capital, Expr.sum(Expr.mulElm(Alpha, X)))         # scalar
        rpen = Expr.mul(capital * risk_lambda, Expr.sum(R))               # scalar
        M.objective("obj", ObjectiveSense.Maximize, Expr.sub(pnl, rpen))

        formulate_time = time.perf_counter() - formulate_start
        logger.info(f"[MP Formulate] N={N}, T={T}, K={K} | Formulate={formulate_time:.4f}s")

        # Return what the caller needs (adjust to your caller; here I return M and X)
        return M, X


    def solve_long_short_portfolio_mp_with_risk_penalty(
        self,
        alphas,                 # list/tuple of (N,) arrays, length T
        covs,                   # list/tuple of (N,N) arrays, length T
        *,
        risk_lambda: float = 1.0,     # scalar, same across periods
        capital: float = 1.0,         # scalar, same across periods
        gme_limit: float = 1.0,       # scalar, same across periods (sum|x| <= gme_limit)
        box: float = 1.0              # -box <= x_i,t <= box
    ) -> np.ndarray:
        """
        Multi-period long/short market-neutral optimizer with risk penalty in the objective.

        maximize   Σ_t capital * ( alpha[t]' x[:,t] )  -  Σ_t capital * risk_lambda * r[t]
        s.t.       1' x[:,t] = 0
                r[t] >= || L[t] x[:,t] ||_2
                Σ_i |x_{i,t}| <= gme_limit
                -box <= x_{i,t} <= box

        Parameters
        ----------
        alphas : list of (N,) np.ndarray
            Expected return vectors per period (absolute values for each period)
        covs : list of (N,N) np.ndarray
            Covariance matrices per period (PSD)
        risk_lambda : float
            Penalty multiplier on portfolio risk magnitude
        capital : float
            Capital scaling for both PnL and risk penalty
        gme_limit : float
            Gross market exposure limit (in weight units)
        box : float
            Per-asset box bound, symmetric

        Returns
        -------
        x_opt : (N, T) np.ndarray
            Optimal weights per period.
        """
        logger = logging.getLogger(__name__)

        # Basic checks
        T = len(alphas)
        if T != len(covs):
            raise ValueError("alphas and covs must have the same length (T).")
        if T == 0:
            return np.array([[]])

        alphas = [np.asarray(a, dtype=float) for a in alphas]
        N = alphas[0].size
        for t, a in enumerate(alphas):
            if a.shape != (N,):
                raise ValueError(f"alpha[{t}] has shape {a.shape}, expected ({N},).")

        covs = [np.asarray(C, dtype=float) for C in covs]
        for t, C in enumerate(covs):
            if C.shape != (N, N):
                raise ValueError(f"cov[{t}] has shape {C.shape}, expected ({N},{N}).")

        # Step 1: Compute Cholesky decompositions
        Ls = self._compute_cholesky_decompositions(covs)

        # Check if we can reuse the model (same config)
        current_config = (N, T, risk_lambda, capital, gme_limit, box)
        can_reuse = (self._mp_model_cache is not None and
                     self._mp_model_config == current_config)

        if can_reuse:
            # Step 2: Reuse existing model, just update parameters
            M = self._mp_model_cache
            x = self._mp_model_x
            L_params = self._mp_model_L_params
            alpha_params = self._mp_model_alpha_params

            # Step 3: Solve with updated parameters and warm start
            x_warmstart = self._mp_last_solution if self._mp_last_solution is not None and self._mp_last_solution.shape == (N, T) else None
            result, solve_time = self._solve_mp_model_reusable(
                M, x, L_params, alpha_params, Ls, alphas, N, T, x_warmstart=x_warmstart
            )

            logger.info(f"[MP Optimizer REUSE] N={N}, T={T} | Solve={solve_time:.4f}s")
        else:
            # Step 2: Build new reusable model
            logger.info(f"[MP Optimizer] Building new reusable model for config: N={N}, T={T}")
            M, x, L_params, alpha_params = self._build_mp_model_reusable(N, T, risk_lambda, capital, gme_limit, box)

            # Cache the model
            self._mp_model_cache = M
            self._mp_model_x = x
            self._mp_model_L_params = L_params
            self._mp_model_alpha_params = alpha_params
            self._mp_model_config = current_config

            # Step 3: Solve with parameters
            x_warmstart = self._mp_last_solution if self._mp_last_solution is not None and self._mp_last_solution.shape == (N, T) else None
            result, solve_time = self._solve_mp_model_reusable(
                M, x, L_params, alpha_params, Ls, alphas, N, T, x_warmstart=x_warmstart
            )

            logger.info(f"[MP Optimizer NEW] N={N}, T={T} | Solve={solve_time:.4f}s")

        # Cache solution for next warm start
        self._mp_last_solution = result.copy()

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

    @staticmethod
    def smooth_alpha_cache(alpha_cache, halflife):
        smoothed_df = alpha_cache.copy()
        alpha = 2**(-1/halflife)
        for col in smoothed_df.columns:
            smoothed_df[col] = smoothed_df[col].ewm(alpha=alpha, adjust=False).mean()

        return smoothed_df


