"""
figure_generation_v3.py

Remakes the manuscript's figures using **both** datasets:
  1) the original theoretical data (JSONs produced by your prior pipeline), and
  2) the new circuit-simulated data (CSVs written by swap_qec_sim.py to data/simulations/).

Conventions
-----------
- Theoretical data are discovered under data/data_streaming/* as JSON files.
- Simulated data are expected at:
    data/simulations/steps_all.csv  (and/or steps_*.csv)
    data/simulations/finals_all.csv (and/or finals_*.csv)

This script unifies fields to canonical columns so that plots can overlay
"Theoretical" and "Simulated" curves/markers with the same legends/labels
as your existing figures.

Notes
-----
* Your purification primitive's post-selected map equals the amplified version's
  conditional output. We therefore overlay simulated points (from the CSVs)
  directly on top of the theoretical curves.
* The theoretical JSON schemas sometimes differ across generators. The loader
  here is tolerant: it looks for multiple synonymous keys (e.g., "noise",
  "noise_type"). Unexpected/unknown records are skipped with a warning.

Outputs
-------
All figures are saved to ./figures (configurable). Filenames are suffixed with
"_v3" to avoid clobbering earlier outputs.
"""

from __future__ import annotations

import json
import math
import os
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Config
# -------------------------
THEORY_BASE = Path("data/data_streaming")
SIM_BASE = Path("data/simulations")
FIG_DIR = Path("figures/results_v2")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_context("paper")
sns.set_style("whitegrid")

# -------------------------
# Helpers: normalization & safe getters
# -------------------------

def _norm_noise(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"depol", "depolarizing", "depolarise", "depolarization"}:
        return "depolarizing"
    if s in {"phase", "dephase_z", "zdephase", "z_dephase", "phaseflip", "phase-flip"}:
        return "dephase_z"
    if s in {"bitflip", "xflip", "x_dephase", "dephase_x", "bit-flip"}:
        return "dephase_x"
    return s


def _first(d: Dict, keys: Iterable[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# -------------------------
# Load THEORETICAL data (JSONs)
# -------------------------

def _read_json_any(path: Path) -> List[Dict]:
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # sometimes wrapped as {"records": [...]} or {"data": [...]}
            for key in ("records", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except Exception as e:
        print(f"Warning: could not parse JSON {path}: {e}")
    return []


def load_theory() -> Dict[str, pd.DataFrame]:
    """Load the various theoretical datasets and coerce to canonical columns.

    Canonical columns used downstream (subset varies per figure):
      M, N, noise, delta, depth, copies_used, fidelity, eps_L, purity,
      physical_error_rate (alias of delta), logical_error (alias of eps_L),
      label (optional series label).
    """
    out: Dict[str, List[Dict]] = {
        "evolution": [],
        "threshold": [],
        "memory_scaling": [],
        "resource_scaling": [],
        "batch_comparison": [],
    }

    if not THEORY_BASE.exists():
        print(f"Warning: theory base {THEORY_BASE} not found — proceeding without theory data.")
        return {k: pd.DataFrame() for k in out}

    # Discover JSONs in known subdirs; be permissive.
    subdirs = {
        "evolution": ["streaming_evolution", "evolution"],
        "threshold": ["threshold_analysis", "streaming_thresholds", "thresholds"],
        "memory_scaling": ["memory_scaling"],
        "resource_scaling": ["resource_scaling", "resources"],
        "batch_comparison": ["batch_comparison", "batch_vs_streaming"],
    }

    def coerce_record(d: Dict) -> Dict:
        noise = _norm_noise(_first(d, ("noise", "noise_type")))
        M = int(_first(d, ("M", "n_qubits", "num_qubits", "register_size"), 0) or 0)
        N = int(_first(d, ("N", "code_size", "copies", "num_copies"), 0) or 0)
        delta = _first(d, ("delta", "physical_error_rate", "phys_err", "p_phys"))
        if delta is not None:
            delta = float(delta)
        depth = _first(d, ("depth", "level", "purification_level"))
        if depth is not None:
            depth = int(depth)
        copies_used = _first(d, ("copies_used",))
        if copies_used is not None:
            copies_used = int(copies_used)
        fidelity = _first(d, ("fidelity", "F", "fid"))
        if fidelity is not None:
            fidelity = float(fidelity)
        eps_L = _first(d, ("eps_L", "logical_error", "epsilon_L", "err_L"))
        if eps_L is not None:
            eps_L = float(eps_L)
        purity = _first(d, ("purity", "Tr_rho2", "trace_rho2"))
        if purity is not None:
            purity = float(purity)
        label = _first(d, ("label",))
        return {
            "M": M,
            "N": N,
            "noise": noise,
            "delta": delta,
            "depth": depth,
            "copies_used": copies_used,
            "fidelity": fidelity,
            "eps_L": eps_L,
            "purity": purity,
            "label": label,
            "_raw": d,
        }

    for key, dirs in subdirs.items():
        recs: List[Dict] = []
        for sub in dirs:
            base = THEORY_BASE / sub
            for fp in map(Path, glob(str(base / "*.json"))):
                recs.extend(_read_json_any(fp))
        if not recs:
            continue
        coerced = [coerce_record(r) for r in recs]
        out[key] = pd.DataFrame(coerced)
        # drop entirely empty cols
        if not out[key].empty:
            out[key] = out[key].drop(columns=[c for c in out[key].columns if out[key][c].isna().all()], errors="ignore")
        print(f"Loaded theory[{key}]: {len(out[key])} rows")

    return out


# -------------------------
# Load SIMULATED data (CSVs)
# -------------------------

def load_sim() -> Dict[str, pd.DataFrame]:
    """Load simulated steps/finals CSVs into canonical columns."""
    steps, finals = pd.DataFrame(), pd.DataFrame()

    steps_all = SIM_BASE / "steps_all.csv"
    finals_all = SIM_BASE / "finals_all.csv"

    # Fallbacks: per-noise files if combined are missing
    if steps_all.exists():
        steps = pd.read_csv(steps_all)
    else:
        parts = [pd.read_csv(p) for p in SIM_BASE.glob("steps_*.csv")] or []
        if parts:
            steps = pd.concat(parts, ignore_index=True)
    if finals_all.exists():
        finals = pd.read_csv(finals_all)
    else:
        parts = [pd.read_csv(p) for p in SIM_BASE.glob("finals_*.csv")] or []
        if parts:
            finals = pd.concat(parts, ignore_index=True)

    if steps.empty and finals.empty:
        print(f"Warning: no simulated CSVs in {SIM_BASE}")

    # Canonicalization / renaming
    def canon_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if "noise" in df.columns:
            df["noise"] = df["noise"].map(_norm_noise)
        if "delta" in df.columns:
            df["delta"] = df["delta"].astype(float)
        for col in ("M", "N", "depth", "copies_used"):
            if col in df.columns:
                df[col] = df[col].astype(int)
        # Derive depth if missing (from copies_used)
        if "depth" not in df.columns and "copies_used" in df.columns:
            df["depth"] = (np.log2(df["copies_used"].replace(0, np.nan))).round().astype("Int64")
        # Standardize column names to match theory
        df = df.rename(columns={
            "fidelity": "fidelity",
            "eps_L": "eps_L",
            "purity": "purity",
        })
        return df

    return {
        "steps": canon_cols(steps),
        "finals": canon_cols(finals),
    }


# -------------------------
# Plotting utilities
# -------------------------

def _add_eps_plot_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a numeric column 'eps_plot' exists.
    Uses 'eps_L' if present; otherwise derives from 'fidelity' assuming depolarizing-family
    (lambda = (D*F-1)/(D-1), eps = (D-1)/D * (1-lambda)). If insufficient info, leaves NaN.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if "eps_L" in df.columns:
        df["eps_plot"] = pd.to_numeric(df["eps_L"], errors="coerce")
        return df
    if {"fidelity", "M"}.issubset(df.columns):
        F = pd.to_numeric(df["fidelity"], errors="coerce")
        M = pd.to_numeric(df["M"], errors="coerce")
        D = (2.0 ** M).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            lam = (D * F - 1.0) / (D - 1.0)
            lam = lam.clip(0.0, 1.0)
            eps = (D - 1.0) / D * (1.0 - lam)
        df["eps_plot"] = eps
    else:
        df["eps_plot"] = np.nan
    return df

def _save(fig: plt.Figure, name: str, fmt: str = "pdf") -> str:
    out = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return str(out)


def _legend(ax: plt.Axes):
    leg = ax.legend(frameon=True, fontsize=9)
    if leg:
        leg.get_frame().set_alpha(0.8)
        leg.get_frame().set_edgecolor("0.8")


# -------------------------
# Figures
# -------------------------

def plot_threshold_curves(theory: Dict[str, pd.DataFrame], sim: Dict[str, pd.DataFrame], fmt: str = "pdf") -> Optional[str]:
    """Logical error (final) vs physical error rate (delta) for multiple code sizes N.
    Overlays theoretical curves with simulated points.
    """
    th = theory.get("threshold", pd.DataFrame())
    sf = sim.get("finals", pd.DataFrame())
    if th.empty and sf.empty:
        print("No data for threshold curves.")
        return None

    # If theory threshold table is missing, try to build from theory evolution/finals-like tables
    if th.empty and not theory.get("evolution", pd.DataFrame()).empty:
        th = theory["evolution"].dropna(subset=["delta", "eps_L"]).groupby(["noise", "M", "N", "delta"]).tail(1)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    colors = sns.color_palette("viridis", 6)

    # Determine noise set from both datasets
    noises = sorted({*_safe_unique(th, "noise"), *_safe_unique(sf, "noise")})
    target_noise = "depolarizing" if "depolarizing" in noises else (noises[0] if noises else None)

    if target_noise is None:
        print("No noise types found for threshold plot.")
        plt.close(fig)
        return None

    # --- Theoretical curves ---
    if not th.empty:
        dth = th[th["noise"] == target_noise]
        for N, sub in dth.groupby("N"):
            sub = sub.sort_values("delta")
            label = f"N={N} — Theoretical"
            ax.plot(sub["delta"], sub["eps_L"], label=label)

    # --- Simulated markers ---
    if not sf.empty:
        dsf = sf[sf["noise"] == target_noise]
        for N, sub in dsf.groupby("N"):
            sub = sub.sort_values("delta")
            label = f"N={N} — Simulated"
            ax.scatter(sub["delta"], sub["eps_L_final"], s=28, marker="o", label=label)

    ax.set_xlabel(r"Physical error rate $\delta$")
    ax.set_ylabel(r"Logical error $\varepsilon_L$")
    ax.set_title(f"Threshold curves — {target_noise}")
    ax.set_yscale("log")
    _legend(ax)
    return _save(fig, f"threshold_curves_{target_noise}", fmt)


def _safe_unique(df: pd.DataFrame, col: str) -> List:
    if df is None or df.empty or col not in df.columns:
        return []
    return sorted([x for x in df[col].dropna().unique().tolist()])


def plot_error_evolution_by_depth(theory: Dict[str, pd.DataFrame], sim: Dict[str, pd.DataFrame], fmt: str = "pdf") -> Optional[str]:
    """Plot eps_L vs purification depth for a few (noise, M, delta) settings.
    Uses theory evolution tables and simulated steps.
    """
    th = theory.get("evolution", pd.DataFrame())
    ss = sim.get("steps", pd.DataFrame())
    th = _add_eps_plot_column(th) if not th.empty else th
    ss = _add_eps_plot_column(ss) if not ss.empty else ss
    if th.empty and ss.empty:
        print("No data for error evolution plot.")
        return None

    # Choose a small panel grid of representative settings found in data
    candidates = []
    for src, df in (("theory", th), ("sim", ss)):
        if df.empty:
            continue
        for noise in _safe_unique(df, "noise"):
            for M in _safe_unique(df, "M"):
                # pick up to 2 deltas present
                for delta in sorted(df[df["M"] == M]["delta"].dropna().unique().tolist())[:2]:
                    candidates.append((noise, M, delta))
    # Deduplicate and take up to 4
    uniq: List[Tuple[str, int, float]] = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)
    uniq = uniq[:4] if uniq else [("depolarizing", 1, 0.1)]

    n = len(uniq)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (noise, M, delta) in zip(axes, uniq):
        # Theoretical
        if not th.empty:
            d = th[(th["noise"] == noise) & (th["M"] == M) & (th["delta"].round(6) == round(delta, 6))]
            if not d.empty:
                g = d.groupby(["N", "depth"]).agg({"eps_plot": "mean"}).reset_index()
                for N, sub in g.groupby("N"):
                    ax.plot(sub["depth"], sub["eps_plot"], label=f"N={N} — Theoretical")
        # Simulated
        if not ss.empty:
            d = ss[(ss["noise"] == noise) & (ss["M"] == M) & (ss["delta"].round(6) == round(delta, 6))]
            if not d.empty:
                g = d.groupby(["N", "depth"]).agg({"eps_L": "last"}).reset_index()
                for N, sub in g.groupby("N"):
                    ax.scatter(sub["depth"], sub["eps_L"], s=28, label=f"N={N} — Simulated")
        ax.set_title(f"{noise}, M={M}, δ={delta}")
        ax.set_xlabel("Purification depth")
        ax.set_yscale("log")
    axes[0].set_ylabel(r"Logical error $\varepsilon_L$")
    _legend(axes[-1])
    return _save(fig, "error_evolution_depth", fmt)


def plot_fidelity_evolution(theory: Dict[str, pd.DataFrame], sim: Dict[str, pd.DataFrame], fmt: str = "pdf") -> Optional[str]:
    th = theory.get("evolution", pd.DataFrame())
    ss = sim.get("steps", pd.DataFrame())
    if th.empty and ss.empty:
        print("No data for fidelity evolution plot.")
        return None

    # Pick a canonical noise and a couple of M values
    noises = sorted({*_safe_unique(th, "noise"), *_safe_unique(ss, "noise")})
    noise = "depolarizing" if "depolarizing" in noises else (noises[0] if noises else None)
    if noise is None:
        return None

    Ms = sorted({*_safe_unique(th, "M"), *_safe_unique(ss, "M")})[:3] or [1]

    fig, axes = plt.subplots(1, len(Ms), figsize=(6*len(Ms), 5), sharey=True)
    if len(Ms) == 1:
        axes = [axes]

    for ax, M in zip(axes, Ms):
        # Theoretical
        if not th.empty:
            d = th[(th["noise"] == noise) & (th["M"] == M)]
            if not d.empty:
                g = d.groupby(["delta", "depth"]).agg({"fidelity": "mean"}).reset_index()
                for delta, sub in g.groupby("delta"):
                    ax.plot(sub["depth"], sub["fidelity"], label=f"δ={delta} — Theoretical")
        # Simulated
        if not ss.empty:
            d = ss[(ss["noise"] == noise) & (ss["M"] == M)]
            if not d.empty:
                g = d.groupby(["delta", "depth"]).agg({"fidelity": "last"}).reset_index()
                for delta, sub in g.groupby("delta"):
                    ax.scatter(sub["depth"], sub["fidelity"], s=28, label=f"δ={delta} — Simulated")
        ax.set_title(f"{noise}, M={M}")
        ax.set_xlabel("Purification depth")
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Fidelity with |ψ⟩")
    _legend(axes[-1])
    return _save(fig, "fidelity_evolution", fmt)


def plot_error_reduction_ratio(theory: Dict[str, pd.DataFrame], sim: Dict[str, pd.DataFrame], fmt: str = "pdf") -> Optional[str]:
    """Plot final / initial logical error as a function of N and δ.
    Uses theory threshold/resource tables if available, and simulated finals.
    """
    th = theory.get("threshold", pd.DataFrame())
    sf = sim.get("finals", pd.DataFrame())
    if th.empty and sf.empty:
        print("No data for error reduction ratio plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Theoretical — try to infer reduction if both initial and final present
    if not th.empty and {"eps_L", "delta", "N"}.issubset(th.columns):
        g = th.groupby(["N", "delta"]).agg({"eps_plot": "mean"}).reset_index()
        for N, sub in g.groupby("N"):
            # Without initial eps_L in this table, just plot eps_L as proxy; overlay simulated ratio separately
            ax.plot(sub["delta"], sub["eps_L"], label=f"N={N} — Theoretical (proxy)")

    # Simulated — exact ratio available
    if not sf.empty and {"error_reduction_ratio", "delta", "N"}.issubset(sf.columns):
        for N, sub in sf.groupby("N"):
            sub = sub.sort_values("delta")
            ax.scatter(sub["delta"], sub["error_reduction_ratio"], s=28, label=f"N={N} — Simulated ratio")

    ax.set_xlabel(r"Physical error rate $\delta$")
    ax.set_ylabel(r"Error reduction: $\varepsilon_L^{\rm final} / \varepsilon_L^{\rm init}$")
    ax.set_yscale("log")
    ax.set_title("Error reduction ratio")
    _legend(ax)
    return _save(fig, "error_reduction_ratio", fmt)


def plot_memory_scaling(theory: Dict[str, pd.DataFrame], sim: Dict[str, pd.DataFrame], fmt: str = "pdf") -> Optional[str]:
    """Max purification depth surrogate vs N. Uses whatever columns exist in THEORY; overlays SIM."""
    th = theory.get("memory_scaling", pd.DataFrame())
    ss = sim.get("steps", pd.DataFrame())
    if th.empty and ss.empty:
        print("No data for memory scaling plot.")
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # THEORY: pick a depth-like column if present
    depth_col = _first_existing_col(th, ["max_depth", "depth", "level", "purification_level", "memory_levels"]) if not th.empty else None
    if not th.empty and depth_col and "N" in th.columns:
        g = th.dropna(subset=["N", depth_col]).groupby("N").agg({depth_col: "max"}).reset_index()
        ax.plot(g["N"], g[depth_col], label="Theoretical", marker="None")

    # SIM: compute max depth per run and then average per N
    if not ss.empty:
        x_sim = _first_existing_col(ss, ["N"]) 
        y_sim = _first_existing_col(ss, ["depth", "level"]) 
        if x_sim and y_sim:
            g = ss.dropna(subset=[x_sim, y_sim]).groupby(["run_id", x_sim]).agg({y_sim: "max"}).reset_index()
            g2 = g.groupby(x_sim).agg({y_sim: "mean"}).reset_index()
            ax.scatter(g2[x_sim], g2[y_sim], label="Simulated", s=30)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Total inputs N")
    ax.set_ylabel("Max purification depth (≈ memory levels)")
    ax.set_title("Memory scaling (O(log N))")
    _legend(ax)
    return _save(fig, "memory_scaling", fmt)


# -------------------------
# Main entry
# -------------------------

def _plot_threshold_curves_fallback(theory: Dict[str, pd.DataFrame], sim: Dict[str, pd.DataFrame], fmt: str = "pdf") -> Optional[str]:
    sf = sim.get("finals", pd.DataFrame())
    if sf.empty:
        print("Fallback threshold: no simulated finals available")
        return None
    df = sf.copy()
    if "delta" not in df.columns and "p_channel" in df.columns:
        if "noise" in df.columns:
            df["noise"] = df["noise"].map(_norm_noise)
            df["delta"] = np.where(df["noise"].eq("depolarizing"), (4.0/3.0)*df["p_channel"], df["p_channel"])
        else:
            df["delta"] = (4.0/3.0) * df["p_channel"]
    if "noise" in df.columns and not df["noise"].dropna().empty:
        noises = df["noise"].dropna().unique().tolist()
        target_noise = "depolarizing" if "depolarizing" in noises else noises[0]
        df = df[df["noise"] == target_noise]
    else:
        target_noise = "unknown"
    ycol = "eps_L_final" if "eps_L_final" in df.columns else ("eps_L" if "eps_L" in df.columns else None)
    if ycol is None:
        print("Fallback threshold: no eps columns in simulated finals")
        return None
    fig, ax = plt.subplots(figsize=(8.5, 7))
    if "N" in df.columns:
        for N, sub in df.groupby("N"):
            sub = sub.dropna(subset=["delta", ycol]).sort_values("delta")
            if sub.empty: continue
            ax.scatter(sub["delta"], sub[ycol], s=28, label=f"N={N} — Simulated")
    else:
        sub = df.dropna(subset=["delta", ycol]).sort_values("delta")
        if not sub.empty:
            ax.scatter(sub["delta"], sub[ycol], s=28, label="Simulated")
    ax.set_xlabel("Physical error rate (delta)")
    ax.set_ylabel("Logical error (epsilon_L)")
    ax.set_title("Threshold curves — " + str(target_noise))
    ax.set_yscale("log")
    _legend(ax)
    return _save(fig, "threshold_curves_" + str(target_noise), fmt)

def generate_all(fmt: str = "pdf") -> Dict[str, Optional[str]]:
    theory = load_theory()
    sim = load_sim()

    outputs: Dict[str, Optional[str]] = {}
    try:
        outputs["threshold_curves"] = plot_threshold_curves(theory, sim, fmt)
    except Exception as e:
        print("plot_threshold_curves failed — using fallback:", e)
        outputs["threshold_curves"] = _plot_threshold_curves_fallback(theory, sim, fmt)
    outputs["error_evolution_depth"] = plot_error_evolution_by_depth(theory, sim, fmt)
    outputs["fidelity_evolution"] = plot_fidelity_evolution(theory, sim, fmt)
    outputs["error_reduction_ratio"] = plot_error_reduction_ratio(theory, sim, fmt)
    outputs["memory_scaling"] = plot_memory_scaling(theory, sim, fmt)

    ok = [k for k, v in outputs.items() if v]
    fail = [k for k, v in outputs.items() if not v]
    print(f"\nGenerated {len(ok)} figures. Failures: {fail}")
    return outputs


if __name__ == "__main__":
    generate_all(fmt="pdf")
