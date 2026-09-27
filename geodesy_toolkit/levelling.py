"""Least squares adjustment of a levelling network, with statistical QA/QC.

Inputs are two CSV files:

    observations.csv   from,to,dh_m,length_km     (height difference to - from)
    fixed.csv          point,height_m             (benchmarks held fixed)

Each observation gets a standard deviation of  sigma_mm_per_km * sqrt(length_km).
The script solves for the unknown heights, then runs:

* the global (chi-square) test on the a posteriori variance factor, and
* Pope's Tau test on every standardised residual, to flag likely blunders.

It also has helpers for geopotential numbers and Helmert orthometric heights.

Usage:
    python -m geodesy_toolkit.levelling observations.csv fixed.csv --sigma 1.0
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Poincare-Prey gravity gradient used for Helmert heights: 0.0424 mGal/m = 4.24e-8 kGal/m
HELMERT_GRADIENT_KGAL_PER_M = 4.24e-8


@dataclass
class Observation:
    frm: str
    to: str
    dh_m: float
    length_km: float


@dataclass
class AdjustmentResult:
    points: list[str]
    heights_m: np.ndarray
    sigma_heights_mm: np.ndarray
    residuals_mm: np.ndarray
    tau: np.ndarray
    tau_critical: float
    dof: int
    variance_factor: float
    chi2: float
    chi2_bounds: tuple[float, float]

    @property
    def global_test_passed(self) -> bool:
        lo, hi = self.chi2_bounds
        return lo <= self.chi2 <= hi

    @property
    def flagged(self) -> np.ndarray:
        """Indices of observations that fail the Tau test."""
        return np.flatnonzero(np.abs(self.tau) > self.tau_critical)


def read_observations(path: str) -> list[Observation]:
    with open(path, newline="") as fh:
        return [Observation(r["from"].strip(), r["to"].strip(), float(r["dh_m"]), float(r["length_km"]))
                for r in csv.DictReader(fh)]


def read_fixed(path: str) -> dict[str, float]:
    with open(path, newline="") as fh:
        return {r["point"].strip(): float(r["height_m"]) for r in csv.DictReader(fh)}


def tau_critical(n_obs: int, dof: int, alpha: float = 0.05) -> float:
    """Critical value of Pope's Tau distribution (Bonferroni over all observations)."""
    if dof < 2:
        return math.inf
    t = stats.t.ppf(1 - alpha / (2 * n_obs), dof - 1)
    return math.sqrt(dof) * t / math.sqrt(dof - 1 + t * t)


def adjust(obs: list[Observation], fixed: dict[str, float], sigma_mm_per_km: float = 1.0,
           alpha: float = 0.05) -> AdjustmentResult:
    """Adjust the network. Heights in metres, standard deviations and residuals in mm."""
    if not fixed:
        raise ValueError("At least one fixed benchmark is needed.")
    points = sorted({p for o in obs for p in (o.frm, o.to)} - fixed.keys())
    idx = {p: i for i, p in enumerate(points)}
    n, u = len(obs), len(points)
    dof = n - u
    if dof < 0:
        raise ValueError(f"Network is under-determined: {n} observations, {u} unknown heights.")

    A = np.zeros((n, u))
    l = np.zeros(n)                      # observed dh reduced by fixed heights, in mm
    sig = np.empty(n)
    for k, o in enumerate(obs):
        rhs = o.dh_m
        if o.to in idx:
            A[k, idx[o.to]] = 1.0
        else:
            rhs -= fixed[o.to]
        if o.frm in idx:
            A[k, idx[o.frm]] = -1.0
        else:
            rhs += fixed[o.frm]
        l[k] = rhs * 1000.0
        sig[k] = sigma_mm_per_km * math.sqrt(max(o.length_km, 1e-6))
    P = np.diag(1.0 / sig ** 2)

    N = A.T @ P @ A
    Ninv = np.linalg.inv(N)
    x = Ninv @ A.T @ P @ l               # heights in mm
    v = A @ x - l                        # residuals in mm
    vtpv = float(v @ P @ v)
    s0_sq = vtpv / dof if dof else float("nan")

    Qvv = np.diag(sig ** 2) - A @ Ninv @ A.T      # cofactor matrix of the residuals
    qvv = np.clip(np.diag(Qvv), 1e-12, None)
    tau = v / (math.sqrt(s0_sq) * np.sqrt(qvv)) if dof else np.zeros(n)

    bounds = (stats.chi2.ppf(alpha / 2, dof), stats.chi2.ppf(1 - alpha / 2, dof)) if dof else (0.0, math.inf)
    return AdjustmentResult(
        points=points,
        heights_m=x / 1000.0,
        sigma_heights_mm=np.sqrt(np.diag(Ninv) * (s0_sq if dof else 1.0)),
        residuals_mm=v,
        tau=tau,
        tau_critical=tau_critical(n, dof, alpha),
        dof=dof,
        variance_factor=s0_sq,
        chi2=vtpv,
        chi2_bounds=bounds,
    )


def geopotential_differences(dh_m, mean_gravity_gal):
    """Geopotential number differences dC = g * dh, in geopotential units (kGal x m)."""
    return np.asarray(mean_gravity_gal, float) * 1e-3 * np.asarray(dh_m, float)


def helmert_height(geopotential_number_gpu: float, surface_gravity_gal: float, tol_m: float = 1e-6) -> float:
    """Helmert orthometric height H = C / (g + 0.0424 mGal/m * H), solved by iteration."""
    g_kgal = surface_gravity_gal * 1e-3
    h = geopotential_number_gpu / g_kgal
    for _ in range(20):
        new = geopotential_number_gpu / (g_kgal + HELMERT_GRADIENT_KGAL_PER_M * h)
        if abs(new - h) < tol_m:
            return new
        h = new
    return h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Least squares adjustment of a levelling network.")
    ap.add_argument("observations", help="CSV with columns from,to,dh_m,length_km")
    ap.add_argument("fixed", help="CSV with columns point,height_m")
    ap.add_argument("--sigma", type=float, default=1.0, help="a priori standard deviation in mm per sqrt(km)")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level for the tests")
    args = ap.parse_args(argv)

    obs = read_observations(args.observations)
    res = adjust(obs, read_fixed(args.fixed), args.sigma, args.alpha)

    print(f"Observations: {len(obs)}   unknown heights: {len(res.points)}   degrees of freedom: {res.dof}")
    print(f"Variance factor: {res.variance_factor:.3f}")
    lo, hi = res.chi2_bounds
    print(f"Global test: chi2 = {res.chi2:.2f}, accepted range [{lo:.2f}, {hi:.2f}] -> "
          f"{'PASSED' if res.global_test_passed else 'FAILED'}")
    print("\nAdjusted heights")
    for p, h, s in zip(res.points, res.heights_m, res.sigma_heights_mm):
        print(f"  {p:12} {h:12.4f} m   +/- {s:6.2f} mm")
    print(f"\nTau test (critical value {res.tau_critical:.2f})")
    flagged = set(res.flagged.tolist())
    for k, o in enumerate(obs):
        mark = "  <-- check" if k in flagged else ""
        print(f"  {o.frm:>8} -> {o.to:<8} v = {res.residuals_mm[k]:7.2f} mm   tau = {res.tau[k]:6.2f}{mark}")
    if not flagged:
        print("  No observation fails the Tau test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
