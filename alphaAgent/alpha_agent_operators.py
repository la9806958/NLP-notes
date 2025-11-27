#!/usr/bin/env python3
"""
AlphaAgent Operator Library
Complete implementation of operators for factor expression language
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Literal
import warnings

warnings.filterwarnings('ignore')

class Operators:
    """Complete operator library for AlphaAgent factor expressions"""
    
    # ============== Core Time-Series (TS) Operators ==============
    
    @staticmethod
    def delay(x: pd.DataFrame, k: int) -> pd.DataFrame:
        """Lag operator: y_t = x_{t-k}"""
        return x.shift(k)
    
    @staticmethod
    def diff(x: pd.DataFrame, k: int) -> pd.DataFrame:
        """Difference operator: y_t = x_t - x_{t-k}"""
        return x - x.shift(k)
    
    @staticmethod
    def ret(p: pd.DataFrame, k: int = 1, mode: str = 'cc') -> pd.DataFrame:
        """Returns calculation
        mode: 'cc' (close-to-close), 'oc' (open-to-close), 'hl' (high-to-low)
        """
        if mode == 'cc':
            return np.log(p / p.shift(k))
        elif mode == 'oc' and k == 1:
            # Assumes we have OHLC data structure
            return np.log(p['close'] / p['open'])
        elif mode == 'hl' and k == 1:
            return np.log(p['high'] / p['low'])
        else:
            raise ValueError(f"Unsupported mode {mode} with k={k}")
    
    @staticmethod
    def ts_mean(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling mean over window w"""
        w = int(w)  # Ensure w is integer
        return x.rolling(window=w, min_periods=w).mean()
    
    @staticmethod
    def ts_std(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling standard deviation over window w"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).std()

    @staticmethod
    def ts_sum(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling sum over window w"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).sum()

    @staticmethod
    def ts_min(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling minimum over window w"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).min()

    @staticmethod
    def ts_max(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling maximum over window w"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).max()

    @staticmethod
    def ts_argmin(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Index of minimum within rolling window"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).apply(lambda arr: w - 1 - np.argmin(arr), raw=True)

    @staticmethod
    def ts_argmax(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Index of maximum within rolling window"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).apply(lambda arr: w - 1 - np.argmax(arr), raw=True)
    
    @staticmethod
    def ema(x: pd.DataFrame, span: int) -> pd.DataFrame:
        """Exponential moving average"""
        return x.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def wma(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Weighted moving average with linear weights"""
        w = int(w)
        weights = np.arange(1, w + 1)
        return x.rolling(window=w, min_periods=w).apply(
            lambda arr: np.average(arr, weights=weights), raw=True
        )

    @staticmethod
    def ts_zscore(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling z-score normalization"""
        w = int(w)
        mean = Operators.ts_mean(x, w)
        std = Operators.ts_std(x, w)
        return (x - mean) / std.replace(0, np.nan)

    @staticmethod
    def ts_pctile(x: pd.DataFrame, w: int, q: float) -> pd.DataFrame:
        """Rolling percentile"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).quantile(q)

    @staticmethod
    def ts_median(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling median over window w"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).median()

    @staticmethod
    def ts_prod(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling product over window w"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).apply(np.prod, raw=True)

    @staticmethod
    def ts_rank(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rank of current value within rolling window (scaled to [0,1])"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).apply(
            lambda arr: (arr[-1] > arr[:-1]).sum() / (len(arr) - 1) if len(arr) > 1 else 0.5,
            raw=True
        )

    @staticmethod
    def corr_ts(x: pd.DataFrame, y: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling correlation between x and y"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).corr(y)

    @staticmethod
    def cov_ts(x: pd.DataFrame, y: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling covariance between x and y"""
        w = int(w)
        return x.rolling(window=w, min_periods=w).cov(y)
    
    @staticmethod
    def beta_ts(y: pd.DataFrame, x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Rolling beta (OLS slope)"""
        cov = Operators.cov_ts(y, x, w)
        var = Operators.ts_std(x, w) ** 2
        return cov / var.replace(0, np.nan)
    
    @staticmethod
    def volatility(p: pd.DataFrame, w: int) -> pd.DataFrame:
        """Price volatility using returns"""
        returns = Operators.ret(p, 1, 'cc')
        return Operators.ts_std(returns, w)
    
    @staticmethod
    def true_range(high: pd.DataFrame, low: pd.DataFrame, close_prev: pd.DataFrame) -> pd.DataFrame:
        """True Range indicator"""
        hl = high - low
        hc = np.abs(high - close_prev)
        lc = np.abs(low - close_prev)
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)
    
    @staticmethod
    def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, w: int) -> pd.DataFrame:
        """Average True Range"""
        close_prev = close.shift(1)
        tr = Operators.true_range(high, low, close_prev)
        return Operators.ts_mean(tr, w)
    
    @staticmethod
    def breakout(x: pd.DataFrame, n: int, direction: str = 'up') -> pd.DataFrame:
        """Breakout indicator"""
        if direction == 'up':
            max_val = Operators.ts_max(Operators.delay(x, 1), n)
            return (x > max_val).astype(np.float16)
        else:
            min_val = Operators.ts_min(Operators.delay(x, 1), n)
            return (x < min_val).astype(np.float16)
    
    @staticmethod
    def persist(cond: pd.DataFrame, w: int) -> pd.DataFrame:
        """Count of True values in last w periods"""
        w = int(w)
        return cond.astype(np.float16).rolling(window=w, min_periods=1).sum()
    
    # ============== Cross-Sectional (CS) Operators ==============
    
    @staticmethod
    def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank (scaled to [0,1])"""
        return x.rank(axis=1, pct=True)
    
    @staticmethod
    def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score normalization"""
        mean = x.mean(axis=1)
        std = x.std(axis=1).replace(0, np.nan)
        return x.sub(mean, axis=0).div(std, axis=0)
    
    @staticmethod
    def cs_winsor(x: pd.DataFrame, p: float) -> pd.DataFrame:
        """Cross-sectional winsorization"""
        lower = x.quantile(p, axis=1)
        upper = x.quantile(1 - p, axis=1)
        return x.clip(lower=lower, upper=upper, axis=0)
    
    @staticmethod
    def cs_neutralize(x: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
        """Neutralize to exposures (e.g., beta, sector)"""
        # Simple implementation - orthogonalize to exposures
        # This would need proper matrix operations for full implementation
        return x - exposures * (x * exposures).sum(axis=1).values[:, None] / (exposures ** 2).sum(axis=1).values[:, None]

    @staticmethod
    def cs_neutralize_multi(x: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
        """Row-wise OLS neutralization: x ⟂ exposures"""
        X = exposures
        XtX_inv = np.linalg.pinv(np.einsum('ij,ik->jk', X.values, X.values))
        betas = (X.values @ XtX_inv) @ np.einsum('ij,ij->i', X.values, x.values)[:, None] / (X.shape[1] or 1)
        # simpler: use projection matrix P = X(X^+) ; x - P x
        P = X.values @ np.linalg.pinv(X.values)
        return pd.DataFrame(x.values - P @ x.values, index=x.index, columns=x.columns)

    @staticmethod
    def cs_rescale(x: pd.DataFrame, norm: str = 'l2') -> pd.DataFrame:
        """Cross-sectional rescaling"""
        if norm == 'l2':
            scale = np.sqrt((x ** 2).sum(axis=1))
        elif norm == 'l1':
            scale = np.abs(x).sum(axis=1)
        elif norm == 'minmax':
            return 2 * (x - x.min(axis=1).values[:, None]) / (x.max(axis=1) - x.min(axis=1)).values[:, None] - 1
        else:
            raise ValueError(f"Unknown norm: {norm}")
        return x.div(scale, axis=0)
    
    # ============== Elementwise/Algebraic/Nonlinear ==============
    
    @staticmethod
    def add(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Element-wise addition"""
        return x + y
    
    @staticmethod
    def sub(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Element-wise subtraction"""
        return x - y
    
    @staticmethod
    def mul(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Element-wise multiplication"""
        return x * y
    
    @staticmethod
    def div(x: pd.DataFrame, y: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        """Safe division"""
        return x / (y + eps)
    
    @staticmethod
    def abs(x: pd.DataFrame) -> pd.DataFrame:
        """Absolute value"""
        return np.abs(x)
    
    @staticmethod
    def sign(x: pd.DataFrame) -> pd.DataFrame:
        """Sign function"""
        return np.sign(x)
    
    @staticmethod
    def clip(x: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
        """Clip values to [a, b]"""
        return x.clip(lower=a, upper=b)
    
    @staticmethod
    def pow(x: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """Power function"""
        return x ** alpha
    
    @staticmethod
    def log1p(x: pd.DataFrame) -> pd.DataFrame:
        """Log(1 + x) for non-negative inputs"""
        return np.log1p(x.clip(lower=0))
    
    @staticmethod
    def sqrt(x: pd.DataFrame) -> pd.DataFrame:
        """Square root"""
        return np.sqrt(x.clip(lower=0))
    
    @staticmethod
    def tanh(x: pd.DataFrame) -> pd.DataFrame:
        """Hyperbolic tangent"""
        return np.tanh(x)
    
    @staticmethod
    def cond(mask: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        """Conditional: a if mask else b"""
        return a.where(mask.astype(bool), b)
    
    @staticmethod
    def gt(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """Greater than"""
        return (x > y).astype(np.float16)
    
    @staticmethod
    def lt(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """Less than"""
        return (x < y).astype(np.float16)
    
    @staticmethod
    def between(x: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
        """Check if x is between a and b"""
        return ((x >= a) & (x <= b)).astype(np.float16)

    @staticmethod
    def neg(x: pd.DataFrame) -> pd.DataFrame:
        """Negation"""
        return -x

    @staticmethod
    def inv(x: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
        """Safe inverse"""
        return 1.0 / (x.replace(0, np.nan) + np.sign(x)*eps)

    @staticmethod
    def emin(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """Element-wise minimum"""
        return pd.concat([x, y if isinstance(y, pd.DataFrame) else x*0 + y], axis=1).min(axis=1).to_frame(x.columns[0]).reindex(columns=x.columns)

    @staticmethod
    def emax(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """Element-wise maximum"""
        return pd.concat([x, y if isinstance(y, pd.DataFrame) else x*0 + y], axis=1).max(axis=1).to_frame(x.columns[0]).reindex(columns=x.columns)

    @staticmethod
    def eq(x, y):
        """Equality comparison"""
        return (x == y).astype(np.float16)

    @staticmethod
    def ne(x, y):
        """Not equal comparison"""
        return (x != y).astype(np.float16)

    @staticmethod
    def ge(x, y):
        """Greater than or equal"""
        return (x >= y).astype(np.float16)

    @staticmethod
    def le(x, y):
        """Less than or equal"""
        return (x <= y).astype(np.float16)

    @staticmethod
    def land(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        """Logical AND"""
        return (a.astype(bool) & b.astype(bool)).astype(np.float16)

    @staticmethod
    def lor(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        """Logical OR"""
        return (a.astype(bool) | b.astype(bool)).astype(np.float16)

    @staticmethod
    def lnot(a: pd.DataFrame) -> pd.DataFrame:
        """Logical NOT"""
        return (~a.astype(bool)).astype(np.float16)

    @staticmethod
    def coalesce(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Return x where not NaN, otherwise y"""
        return x.where(x.notna(), y)

    @staticmethod
    def signed_pow(x: pd.DataFrame, a: float) -> pd.DataFrame:
        """Signed power function"""
        return np.sign(x) * (np.abs(x) ** a)

    @staticmethod
    def safe_log(x: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
        """Safe logarithm"""
        return np.log(x.clip(lower=eps))

    @staticmethod
    def pct_change(p: pd.DataFrame, k: int = 1) -> pd.DataFrame:
        """Percentage change"""
        return p.pct_change(k)

    # ============== Price/Volume Specific ==============
    
    @staticmethod
    def hl_range(high: pd.DataFrame, low: pd.DataFrame, close_prev: pd.DataFrame) -> pd.DataFrame:
        """Normalized high-low range"""
        return (high - low) / close_prev
    
    @staticmethod
    def turnover(volume: pd.DataFrame, w: int) -> pd.DataFrame:
        """Volume turnover ratio"""
        adv = Operators.ts_mean(volume, w)
        return volume / adv.replace(0, np.nan)
    
    @staticmethod
    def adv(volume: pd.DataFrame, w: int) -> pd.DataFrame:
        """Average daily/period volume"""
        return Operators.ts_mean(volume, w)
    
    # ============== Microstructure (if available) ==============
    
    @staticmethod
    def spread(ask: pd.DataFrame, bid: pd.DataFrame) -> pd.DataFrame:
        """Bid-ask spread"""
        return ask - bid
    
    @staticmethod
    def rel_spread(ask: pd.DataFrame, bid: pd.DataFrame) -> pd.DataFrame:
        """Relative spread"""
        mid = (ask + bid) / 2
        return (ask - bid) / mid
    
    @staticmethod
    def order_imb(bid_sz: pd.DataFrame, ask_sz: pd.DataFrame) -> pd.DataFrame:
        """Order imbalance"""
        return (bid_sz - ask_sz) / (bid_sz + ask_sz).replace(0, np.nan)
    
    @staticmethod
    def amihud(returns: pd.DataFrame, volume: pd.DataFrame, w: int) -> pd.DataFrame:
        """Amihud illiquidity measure"""
        abs_ret = np.abs(returns)
        return Operators.ts_mean(abs_ret / volume.replace(0, np.nan), w)
    
    # ============== Event/Pattern Helpers ==============
    
    @staticmethod
    def crossover(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Up-cross: x crosses above y"""
        return ((x.shift(1) <= y.shift(1)) & (x > y)).astype(np.float16)
    
    @staticmethod
    def crossunder(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Down-cross: x crosses below y"""
        return ((x.shift(1) >= y.shift(1)) & (x < y)).astype(np.float16)
    
    # ============== Signal Hygiene ==============
    
    @staticmethod
    def demean_ts(x: pd.DataFrame, w: int) -> pd.DataFrame:
        """Remove rolling mean"""
        return x - Operators.ts_mean(x, w)
    
    @staticmethod
    def debias_cs(x: pd.DataFrame) -> pd.DataFrame:
        """Remove cross-sectional mean"""
        return x.sub(x.mean(axis=1), axis=0)
    
    @staticmethod
    def smooth_ts(x: pd.DataFrame, w: int, method: str = 'ema') -> pd.DataFrame:
        """Smooth time series"""
        w = int(w)
        if method == 'ema':
            return Operators.ema(x, w)
        elif method == 'median':
            return x.rolling(window=w, min_periods=w).median()
        else:
            return Operators.ts_mean(x, w)
    
    @staticmethod
    def fillna_ts(x: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """Fill NaN values (no lookahead - only ffill or constant fill)"""
        if method == 'ffill':
            return x.fillna(method='ffill')
        else:
            return x.fillna(0)