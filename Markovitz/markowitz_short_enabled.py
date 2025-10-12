#!/usr/bin/env python3
"""
Markowitz Portfolio Optimization — Short-Selling Enabled (Yahoo Finance)
========================================================================

Features
--------
- Fetches Adjusted Close prices from Yahoo Finance for any tickers.
- Computes daily log returns; annualizes mean & covariance.
- Solves THREE portfolios with **shorting allowed**:
  1) GMV (Global Minimum Variance)
  2) MSR (Max Sharpe / tangency)
  3) Target-return efficient portfolio
- Constraint set is configurable:
  * Bounds per weight: [min_w, max_w]  (default: [-1.0, 2.0])
  * Net exposure: sum(w) = S  (default S = 1.0; set S > 1 for net leverage)
- Saves CSV tables and a PNG chart of the frontier.

Usage
-----
pip install yfinance pandas numpy scipy matplotlib
python markowitz_short_enabled.py \
  --tickers SPY QQQ TLT GLD \
  --start 2018-01-01 --end 2025-10-01 \
  --rf 0.03 --target 0.10 \
  --min_weight -1.0 --max_weight 2.0 \
  --net_sum 1.0

Notes
-----
- Allowing shorting (negative weights) can dramatically increase risk.
- With sum(w)=1, short proceeds fund longs (classic long/short, net 1.0).
- To allow net leverage (borrowing cash), set --net_sum > 1 (e.g., 1.2).
"""

import argparse
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Only needed when running locally with internet
try:
    import yfinance as yf
except Exception:
    print("yfinance is required. Install with: pip install yfinance")
    raise

from scipy.optimize import minimize


# ---------------------- CLI ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Markowitz optimization with short-selling enabled.")
    p.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "TLT", "GLD", "IWM", "EFA"],
        help="Space-separated list of Yahoo tickers.",
    )
    p.add_argument("--start", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--rf", type=float, default=0.03, help="Annual risk-free rate (e.g., 0.03 = 3%)")
    p.add_argument(
        "--target", type=float, default=None, help="Optional target annual return for an efficient portfolio"
    )
    p.add_argument("--min_weight", type=float, default=-1.0, help="Lower bound for each weight (allow shorts)")
    p.add_argument("--max_weight", type=float, default=2.0, help="Upper bound for each weight (allow leverage)")
    p.add_argument("--net_sum", type=float, default=1.0, help="Sum of weights constraint (net exposure).")
    p.add_argument("--frontier_points", type=int, default=80, help="Number of points on efficient frontier sweep.")
    p.add_argument("--outdir", type=str, default="Markovitz/outputs_markowitz_short", help="Output directory")
    return p.parse_args()


# ------------------ Data & Returns ------------------
def load_prices(tickers: Sequence[str], start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="any")


def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def annualize_stats(returns: pd.DataFrame, periods_per_year: int = 252) -> Tuple[np.ndarray, np.ndarray]:
    mu = returns.mean().values * periods_per_year
    Sigma = returns.cov().values * periods_per_year
    return mu, Sigma


# ------------------ Portfolio math ------------------
def port_stats(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> Tuple[float, float]:
    r = float(w @ mu)
    v = float(w @ Sigma @ w)
    return r, sqrt(v)


def _solve_slsqp(obj, x0, cons, bounds):
    res = minimize(obj, x0=x0, constraints=cons, bounds=bounds, method="SLSQP")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def solve_gmv(mu: np.ndarray, Sigma: np.ndarray, net_sum: float, bounds: tuple[float, float]):
    n = len(mu)
    x0 = np.full(n, net_sum / n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - net_sum},)
    bnds = tuple([bounds] * n)
    return _solve_slsqp(lambda w: w @ Sigma @ w, x0, cons, bnds)


def solve_msr(mu: np.ndarray, Sigma: np.ndarray, rf: float, net_sum: float, bounds: tuple[float, float]):
    n = len(mu)
    x0 = np.full(n, net_sum / n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - net_sum},)
    bnds = tuple([bounds] * n)

    def neg_sharpe(w):
        r, s = port_stats(w, mu, Sigma)
        return -(r - rf) / s if s > 0 else 1e9

    return _solve_slsqp(neg_sharpe, x0, cons, bnds)


def solve_target_return(mu: np.ndarray, Sigma: np.ndarray, R: float, net_sum: float, bounds: tuple[float, float]):
    n = len(mu)
    x0 = np.full(n, net_sum / n)
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - net_sum},
        {"type": "eq", "fun": lambda w, mu=mu, R=R: float(w @ mu) - R},
    )
    bnds = tuple([bounds] * n)
    try:
        return _solve_slsqp(lambda w: w @ Sigma @ w, x0, cons, bnds)
    except RuntimeError:
        # Relax exact target to inequality by penalty if infeasible under bounds
        cons_relaxed = ({"type": "eq", "fun": lambda w: np.sum(w) - net_sum},)

        def obj(w):
            var = w @ Sigma @ w
            shortfall = max(0.0, R - float(w @ mu))
            return var + 1e4 * shortfall**2

        return _solve_slsqp(obj, x0, cons_relaxed, bnds)


def efficient_frontier(
    mu: np.ndarray, Sigma: np.ndarray, k: int, net_sum: float, bounds: tuple[float, float]
) -> pd.DataFrame:
    R_min, R_max = float(min(mu)), float(max(mu))
    targets = np.linspace(R_min + 1e-4, R_max - 1e-4, k)
    recs = []
    for R in targets:
        try:
            w = solve_target_return(mu, Sigma, R, net_sum, bounds)
            r, s = port_stats(w, mu, Sigma)
            recs.append({"risk": s, "return": r, "R_target": R, "weights": w})
        except Exception:
            pass
    return pd.DataFrame(recs)


# ------------------ Pretty tables ------------------
def weights_table(
    name: str, tickers: Sequence[str], w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float
) -> pd.DataFrame:
    r, s = port_stats(w, mu, Sigma)
    df = pd.DataFrame({"Ticker": tickers, "Weight": w})
    df.loc[len(df)] = ["—", np.nan]
    df.loc[len(df)] = ["Portfolio Return (ann.)", r]
    df.loc[len(df)] = ["Portfolio Risk σ (ann.)", s]
    df.loc[len(df)] = ["Sharpe (excess/σ)", (r - rf) / s]
    df.insert(0, "Portfolio", name)
    return df


# ------------------ Main ------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    end = args.end or datetime.today().strftime("%Y-%m-%d")

    print(f"Downloading prices for: {args.tickers}  [{args.start} → {end}]")
    prices = load_prices(args.tickers, args.start, end)
    if prices.empty:
        raise SystemExit("No price data. Try different tickers/dates.")

    rets = daily_log_returns(prices)
    mu, Sigma = annualize_stats(rets)

    bounds = (args.min_weight, args.max_weight)

    # Solve key portfolios (shorting enabled via bounds)
    w_gmv = solve_gmv(mu, Sigma, net_sum=args.net_sum, bounds=bounds)
    w_msr = solve_msr(mu, Sigma, rf=args.rf, net_sum=args.net_sum, bounds=bounds)
    target = args.target if args.target is not None else float(np.median(mu))
    w_tgt = solve_target_return(mu, Sigma, R=target, net_sum=args.net_sum, bounds=bounds)

    # Save weights tables
    tables = [
        weights_table("GMV (min-variance)", prices.columns, w_gmv, mu, Sigma, args.rf),
        weights_table("MSR (max Sharpe)", prices.columns, w_msr, mu, Sigma, args.rf),
        weights_table(f"Target Return ≈ {target:.2%}", prices.columns, w_tgt, mu, Sigma, args.rf),
    ]
    weights_summary = pd.concat(tables, ignore_index=True)
    weights_summary["Weight"] = weights_summary["Weight"].astype(float).round(6)
    weights_path = outdir / "weights_summary.csv"
    weights_summary.to_csv(weights_path, index=False)
    print(f"Saved: {weights_path}")

    # Efficient frontier
    ef = efficient_frontier(mu, Sigma, k=args.frontier_points, net_sum=args.net_sum, bounds=bounds)
    if not ef.empty:
        W = np.vstack(ef["weights"].to_list())
        ef_expanded = ef.drop(columns=["weights"]).copy()
        for i, t in enumerate(prices.columns):
            ef_expanded[f"w_{t}"] = W[:, i]
        ef_path = outdir / "efficient_frontier.csv"
        ef_expanded.to_csv(ef_path, index=False)
        print(f"Saved: {ef_path}")

    # Plot
    plt.figure()
    if not ef.empty:
        plt.plot(ef["risk"], ef["return"], label="Efficient Frontier")
    # Individual assets
    vol = np.sqrt(np.diag(Sigma))
    plt.scatter(vol, mu, marker="x", label="Assets")
    for i, t in enumerate(prices.columns):
        plt.annotate(str(t), (vol[i], mu[i]))
    # Highlight GMV / MSR
    for name, w, marker in [("GMV", w_gmv, "o"), ("MSR", w_msr, "s")]:
        r, s = port_stats(w, mu, Sigma)
        plt.scatter([s], [r], marker=marker, s=60, label=name)
    plt.xlabel("Risk (stdev, annualized)")
    plt.ylabel("Return (annualized)")
    plt.title("Markowitz Efficient Frontier — Shorting Enabled")
    plt.legend()
    plt.tight_layout()
    fig_path = outdir / "efficient_frontier.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")

    # Summary
    with open(outdir / "summary.txt", "w") as f:
        f.write("Markowitz (Short-Enabled) — Summary\n")
        f.write(f"Tickers: {', '.join(args.tickers)}\n")
        f.write(f"Date range: {args.start} → {end}\n")
        f.write(f"Risk-free (annual): {args.rf:.2%}\n")
        f.write(f"Bounds per weight: [{args.min_weight}, {args.max_weight}]\n")
        f.write(f"Net exposure sum(w): {args.net_sum}\n\n")
        f.write("GMV and MSR and Target weights are in weights_summary.csv\n")

    print("Done. See outputs in:", outdir)


if __name__ == "__main__":
    main()
