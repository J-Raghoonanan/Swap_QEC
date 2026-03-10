"""
Analytic theory plots for PQEC (purification-based quantum error correction).

IMPORTANT — VD CONSISTENCY:
  The purification update rule used throughout this file is the VD map:

      ρ → ρ²/Tr(ρ²)

  For a single qubit (D=2) this gives the Bloch-sphere radial update:

      r → P(r) = 2r / (1 + r²)

  This replaces the old SWAP map P(r) = 4r/(3+r²) used in an earlier version.
  All fixed-point formulas, thresholds, and F₀ expressions are updated to
  match.  See manuscript Eq. (gen_error_reduction) and the \Jon{} boxes.

Generates figures:
  (1) F_out vs F (isotropic family, VD map)
  (2) Error evolution with general bounds
  (5) GHZ per-round error ratio vs coherence gamma
  (7) Dephasing anisotropy vs isotropy
  (8) Round-count scaling n_* vs M for GHZ
  (9) F_out vs dimension D at fixed F
  Comprehensive 5-panel grid (panels a–e)

Saves PDFs under: figures/theory_ana_plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# ---------------------------------------------------------------------
# Publication-quality plotting
# ---------------------------------------------------------------------
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 30,
    'axes.labelsize': 25,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 12
})

MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'h', '*']
def _mk(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


# ---------------------------------------------------------------------
# Analytic helpers
# ---------------------------------------------------------------------

def T_isotropic(F: float, D: int) -> float:
    """Register purity for the isotropic/Werner family."""
    return F**2 + (1 - F)**2 / (D - 1)

def Fout_isotropic(F: np.ndarray, D: int) -> np.ndarray:
    """VD fidelity map for Werner/isotropic family (manuscript \Jon{} box):
        F' = F² / (F² + (1-F)²/(D-1))
    """
    numerator   = F * F
    denominator = F**2 + (1.0 - F)**2 / (D - 1)
    return numerator / denominator

def err_bounds(F: float, D: int) -> Tuple[float, float]:
    """Lower/upper bounds on next-round error (isotropic case)."""
    T = T_isotropic(F, D)
    lower = (1 + T - 2*F) / (1 + T)
    upper = (1 + T - F - F**2) / (1 + T)
    return lower, upper

def ghz_gamma_update(gamma: float) -> float:
    """GHZ coherence update: gamma -> 4*gamma/(3+gamma²)."""
    return (4.0 * gamma) / (3.0 + gamma*gamma)

def ghz_err_ratio_per_round(gamma: np.ndarray) -> np.ndarray:
    """Exact per-round error ratio for GHZ: eps_out/eps = (3-gamma)/(3+gamma²)."""
    return (3.0 - gamma) / (3.0 + gamma*gamma)

def alpha_from_betaz(beta_z: float) -> float:
    """Isotropized contraction factor alpha = (1+2*beta_z)/3."""
    return (1.0 + 2.0 * beta_z) / 3.0

def n_star_from_gamma0(gamma0: float) -> float:
    """Rounds n_* to lift small gamma0 to O(1): n_* ~ log(1/gamma0)/log(4/3)."""
    return np.log(1.0 / gamma0) / np.log(4.0 / 3.0)

def bloch_update_radius_vd(r: float) -> float:
    """VD radial update for a single qubit (D=2): r → 2r/(1+r²).

    Derived from F' = F²/(F²+(1-F)²) with F=(1+r)/2:
        F' = (1+r)²/(2(1+r²))
        r' = 2F'-1 = 2r/(1+r²)

    Old SWAP map was r → 4r/(3+r²); threshold shifts from p≤1/4 to p≤1/2.
    """
    return 2.0 * r / (1.0 + r * r)


# ---------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------

class AnalyticTheoryPlotter:
    def __init__(self, figures_dir: str = "figures/theory_ana_plots"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # (1) F_out vs F, isotropic/Werner family
    def plot_fout_vs_f_isotropic(self,
                                 D_list: Optional[List[int]] = None,
                                 save_format: str = "pdf") -> str:
        if D_list is None:
            D_list = [2, 4, 32]
        F = np.linspace(0.0, 1.0, 500)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["red", "green", "blue", "orange", "purple"]

        for i, D in enumerate(D_list):
            Fout = Fout_isotropic(F, D)
            ax.plot(F, Fout, marker=_mk(i), color=colors[i], label=f'D={D}',
                    markevery=50, markersize=12)
        ax.plot(F, F, '--', color='gray', linewidth=2, alpha=0.7, label='Identity')

        ax.set_xlabel(r'Input Fidelity, $F$', fontsize=40)
        ax.set_ylabel(r"Output Fidelity, $F'$", fontsize=40)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=22, frameon=False)
        plt.tight_layout()

        filename = f"fout_vs_f_isotropic.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # (2) Error evolution with bounds (isotropic recursion)
    def plot_error_evolution_with_bounds(self,
                                         F0: float = 0.7,
                                         D: int = 8,
                                         n_steps: int = 12,
                                         save_format: str = "pdf") -> str:
        F_exact = [F0]
        for _ in range(n_steps):
            F_exact.append(Fout_isotropic(np.array([F_exact[-1]]), D)[0])
        F_exact = np.array(F_exact)
        eps_exact = 1.0 - F_exact

        eps_lower, eps_upper = [], []
        for F in F_exact[:-1]:
            low, up = err_bounds(F, D)
            eps_lower.append(low); eps_upper.append(up)
        eps_lower = np.array(eps_lower)
        eps_upper = np.array(eps_upper)

        fig, ax = plt.subplots(figsize=(10, 8))
        n = np.arange(len(eps_exact))
        v = plt.cm.viridis([0.15, 0.55, 0.85])

        ax.semilogy(n, eps_exact, '-', marker=_mk(0), color=v[2],
                    label='Exact error', linewidth=3)
        ax.semilogy(n[1:], eps_lower, '--', marker=_mk(1), color=v[0],
                    label='Lower bound', alpha=0.95)
        ax.semilogy(n[1:], eps_upper, '--', marker=_mk(2), color=v[0],
                    label='Upper bound', alpha=0.95)

        ax.set_xlabel(r'Purification Level, $n$', fontsize=25)
        ax.set_ylabel(r'Error, $\varepsilon_n$', fontsize=25)
        ax.set_title(r'Error Decay with Bounds (Isotropic Family)', fontsize=30)
        ax.legend(loc='best', fontsize=14)
        ax.set_ylim(1e-6, 1.0)
        plt.tight_layout()

        filename = f"error_evolution_with_bounds_D{D}_F0_{F0:.2f}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # (5) GHZ per-round error ratio vs gamma
    def plot_ghz_error_ratio(self, save_format: str = "pdf") -> str:
        gamma = np.linspace(0.0, 1.0, 500)
        ratio = ghz_err_ratio_per_round(gamma)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(gamma, ratio, '-', color=plt.cm.viridis(0.65),
                label=r'$\varepsilon_{\mathrm{out}}/\varepsilon$')
        for i, g in enumerate([0.1, 0.5, 0.9]):
            ax.plot([g], [ghz_err_ratio_per_round(np.array([g]))[0]],
                    marker=_mk(i), color=plt.cm.viridis(0.65))

        ax.set_xlabel(r'GHZ Coherence, $\gamma$', fontsize=25)
        ax.set_ylabel(r'Per-round Error Ratio, $\varepsilon_{\mathrm{out}}/\varepsilon$', fontsize=25)
        ax.set_title(r'GHZ Per-round Error Reduction (Exact)', fontsize=30)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()

        filename = f"ghz_error_ratio_per_round.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # (7) Dephasing anisotropy vs isotropy (single qubit, equatorial target)
    def plot_anisotropy_vs_isotropy_single_qubit(self,
                                                 beta_z: float = 0.5,
                                                 n_steps: int = 12,
                                                 save_format: str = "pdf") -> str:
        r_no = [beta_z]
        for _ in range(n_steps):
            r_no.append(bloch_update_radius_vd(r_no[-1]))
        r_no = np.array(r_no)

        alpha = alpha_from_betaz(beta_z)
        r_iso = [alpha]
        for _ in range(n_steps):
            r_iso.append(bloch_update_radius_vd(r_iso[-1]))
        r_iso = np.array(r_iso)

        F_no  = 0.5 * (1.0 + r_no)
        F_iso = 0.5 * (1.0 + r_iso)

        fig, ax = plt.subplots(figsize=(10, 8))
        n = np.arange(len(F_no))
        base_color = plt.cm.viridis(0.7)

        ax.plot(n, F_no, '-', marker=_mk(0), color=base_color,
                label=rf'No twirl ($\beta_z={beta_z:.2f}$)')
        ax.plot(n, F_iso, '--', marker=_mk(1), color=base_color,
                label=fr'Isotropized ($\alpha={alpha:.2f}$)')

        ax.set_xlabel(r'Purification Level, $n$', fontsize=25)
        ax.set_ylabel(r'Fidelity, $F_n$', fontsize=25)
        ax.set_title(r'Dephasing Anisotropy vs Isotropy (Single Qubit)', fontsize=30)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()

        filename = f"anisotropy_vs_isotropy_qubit_betaz_{beta_z:.2f}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # (8) Round-count scaling n_* vs M for GHZ
    def plot_round_count_vs_M(self,
                              beta_z_list: Optional[List[float]] = None,
                              M_list: Optional[List[int]] = None,
                              save_format: str = "pdf") -> str:
        if beta_z_list is None:
            beta_z_list = [0.6, 0.8, 0.9]
        if M_list is None:
            M_list = list(range(2, 65, 2))

        fig, ax = plt.subplots(figsize=(10, 8))
        v_positions = np.linspace(0.0, 1.0, len(beta_z_list))
        v_colors = [plt.cm.viridis(pos) for pos in v_positions]

        for i, beta_z in enumerate(beta_z_list):
            alpha = alpha_from_betaz(beta_z)
            color_i = v_colors[i]

            n_no, n_iso = [], []
            for M in M_list:
                n_no.append(n_star_from_gamma0(beta_z**M))
                n_iso.append(n_star_from_gamma0(alpha**M))

            ax.plot(M_list, n_no, '-', marker=_mk(2*i), color=color_i,
                    label=rf'No twirl ($\beta_z={beta_z:.2f}$)')
            ax.plot(M_list, n_iso, '--', marker=_mk(2*i+1), color=color_i,
                    label=rf'Isotropized ($\alpha={alpha:.2f}$)')

        ax.set_xlabel(r'System Size, $M$ (qubits)', fontsize=25)
        ax.set_ylabel(r'Rounds to Constant Coherence, $n_\star$', fontsize=25)
        ax.set_title(r'GHZ Round-count Scaling', fontsize=30)
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()

        filename = f"round_count_vs_M.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # (9) F_out vs D at fixed F
    def plot_fout_vs_D_fixedF(self,
                              F0: float = 0.7,
                              D_list: Optional[List[int]] = None,
                              save_format: str = "pdf") -> str:
        if D_list is None:
            D_list = [2, 4, 8, 16, 32, 64, 128]
        Fout_vals = [Fout_isotropic(np.array([F0]), D)[0] for D in D_list]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(D_list, Fout_vals, '-o', color=plt.cm.viridis(0.75),
                label=rf'$F_0={F0:.2f}$')
        ax.set_xscale('log', base=2)

        ax.set_xlabel(r'Dimension, $D$', fontsize=25)
        ax.set_ylabel(r'Output Fidelity, $F_{\mathrm{out}}$', fontsize=25)
        ax.set_title(r'VD Map: $F_{\mathrm{out}}$ vs $D$ at Fixed $F$', fontsize=30)
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()

        filename = f"fout_vs_D_fixedF_{F0:.2f}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Single-qubit VD radial dynamics  (used by ALL five panels)
    # ------------------------------------------------------------------

    def P(self, x: np.ndarray) -> np.ndarray:
        """One VD radial update (D=2): r → 2r/(1+r²).

        This is the Bloch-sphere form of the VD map F' = F²/(F²+(1-F)²).
        Old SWAP map was P(r)=4r/(3+r²) — do NOT use that here.
        """
        return 2.0 * x / (1.0 + x**2)

    def iterate_r(self, r0: float, p: float, ell: int, n_iter: int) -> np.ndarray:
        """Iterate r_{n+1} = P^(ell)((1-p) r_n) for n_iter steps.

        Noise per cycle: r → (1-p)r  (global depolarizing, D=2).
        Purification per cycle: apply P exactly ell times.
        """
        r = float(r0)
        traj = [r]
        for _ in range(n_iter):
            x = (1.0 - p) * r          # noise: Werner mixing contracts Bloch sphere
            for _ in range(ell):        # ell VD rounds
                x = self.P(x)
            r = x
            traj.append(r)
        return np.array(traj)

    def plot_r_vs_iteration_multi_r0(
        self,
        p: float = 0.1,
        ell_list=(1, 2),
        r0_list=(0.1, 0.5, 1.0),
        n_iter: int = 30,
        save_format: str = "pdf",
    ):
        plt.style.use("seaborn-v0_8-paper")
        colors = ["red", "green", "blue", "orange", "purple"]

        for ell in ell_list:
            fig, ax = plt.subplots(figsize=(10, 8))

            for i, r0 in enumerate(r0_list):
                traj = self.iterate_r(r0=r0, p=p, ell=ell, n_iter=n_iter)
                it = np.arange(len(traj))
                ax.plot(it, traj,
                        marker=_mk(i), color=colors[i % len(colors)],
                        label=fr"$r_0={r0}$",
                        markevery=max(1, len(it) // 10),
                        markersize=12, linewidth=2)

            ax.set_title(fr"$\ell={ell},\; p={p}$", fontsize=40)
            ax.set_xlabel("Cycle", fontsize=40)
            ax.set_ylabel(r"Bloch radius, $r=|\vec r|$", fontsize=40)
            ax.set_xlim(0, n_iter); ax.set_ylim(0, 1.05)
            ax.set_xticks([0, n_iter//4, n_iter//2, 3*n_iter//4, n_iter])
            ax.tick_params(axis="both", which="major", labelsize=26, length=8, width=2)
            ax.legend(loc="lower right", fontsize=16, frameon=True, framealpha=0.9)

            plt.tight_layout()
            filename = f"r_vs_iteration_p{p}_ell{ell}_multi_r0.{save_format}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight"); plt.close()
            print(f"Saved {filename}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Fixed-point Bloch radii  (updated for VD)
    # ------------------------------------------------------------------

    def rfix_ell1(self, p: np.ndarray) -> np.ndarray:
        """VD fixed-point radius for ell=1.

        Solves P((1-p)r) = r with P(x) = 2x/(1+x²):
            r² = (1-2p) / (1-p)²

        Threshold: p_th = 1/2  (was 1/4 under the old SWAP map).
        """
        p = np.asarray(p, dtype=float)
        num = 1.0 - 2.0 * p
        den = (1.0 - p) ** 2
        return np.sqrt(np.where(num > 0.0, num / den, 0.0))

    def rfix_ell2(self, p: np.ndarray) -> np.ndarray:
        """VD fixed-point radius for ell=2.

        Solves P²((1-p)r) = r with P(x) = 2x/(1+x²):
            r² = (2√(1+p²) - (1+2p)) / (1-p)²

        Threshold: p_th = 3/4  (was ≈0.4375 under the old SWAP map).

        Derivation:
            Let a=(1-p)r.  P²(a) = 4a(1+a²)/(a⁴+6a²+1).
            Setting P²(a)/(1-p) = r = a/(1-p) gives the quartic
            a⁴ + (2+4p)a² + (4p-3) = 0,
            whose positive root is a² = -(1+2p) + 2√(1+p²).
        """
        p = np.asarray(p, dtype=float)
        u = 2.0 * np.sqrt(1.0 + p ** 2) - (1.0 + 2.0 * p)
        return np.sqrt(np.where(u > 0.0, u / (1.0 - p) ** 2, 0.0))

    def rfix_general(self, p: np.ndarray, ell: int) -> np.ndarray:
        """VD fixed-point radius for arbitrary ell (D=2).

        Uses closed-form for ell=0,1,2; binary-search for ell≥3.
        """
        p = np.asarray(p, dtype=float)

        if ell == 0:
            return np.zeros_like(p)
        if ell == 1:
            return self.rfix_ell1(p)
        if ell == 2:
            return self.rfix_ell2(p)

        # Higher ell: binary search
        r_fix = np.zeros_like(p)
        for i, p_val in enumerate(p):
            if p_val >= 1.0:
                continue
            r_low, r_high = 0.0, 1.0
            for _ in range(50):
                r_test = (r_low + r_high) / 2.0
                r_next = self.iterate_r(r_test, p_val, ell, 1)[1]
                if abs(r_next - r_test) < 1e-10:
                    r_fix[i] = r_test
                    break
                elif r_next > r_test:
                    r_low = r_test
                else:
                    r_high = r_test
            else:
                r_fix[i] = r_test
        return r_fix

    def calculate_gamma_first_drop(self, r0: float, p: float, ell: int) -> float:
        """γ = F(t=0) - F(t=1): fidelity drop after the first cycle."""
        traj = self.iterate_r(r0, p, ell, 1)
        F0 = (1.0 + traj[0]) / 2.0
        F1 = (1.0 + traj[1]) / 2.0
        return max(0.0, F0 - F1)

    def plot_rfix_vs_p(self, figures_dir, p_max=1.0, n=500, save_format="pdf"):
        plt.style.use("seaborn-v0_8-paper")
        colors = ["red", "blue"]

        p = np.linspace(0.0, p_max, n)
        r1 = self.rfix_ell1(p)
        r2 = self.rfix_ell2(p)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(p, r1, color=colors[0], linewidth=3, marker="o", markevery=50,
                markersize=10, label=r"$\ell=1$")
        ax.plot(p, r2, color=colors[1], linewidth=3, marker="s", markevery=50,
                markersize=10, label=r"$\ell=2$")

        ax.set_xlim(0, p_max); ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"Physical Error Rate, $p$", fontsize=40)
        ax.set_ylabel(r"$r_{\mathrm{fix}}$", fontsize=40)
        ax.tick_params(axis="both", which="major", labelsize=24, length=8, width=2)
        ax.legend(loc="lower left", fontsize=18, frameon=True)

        plt.tight_layout()
        out = self.figures_dir / f"rfix_vs_p.{save_format}"
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        print(f"Saved {out}")
        return str(out)

    # ------------------------------------------------------------------
    # F₀ analytical  (updated for VD)
    # ------------------------------------------------------------------

    def F0_analytical(self, p: np.ndarray, D: int = 2) -> np.ndarray:
        """Steady-state fidelity F₀ for ell=1 VD under global depolarizing.

        From manuscript (\Jon{} box):
            F₀ = (1/2)(1 + √(1 − 4(D−1)p² / (D²(1−p)²)))

        For D=2 this simplifies to:
            F₀ = (1/2)(1 + √(1 − p²/(1−p)²))  =  (1/2)(1 + √(1−2p)/(1−p))

        Valid when the discriminant ≥ 0:
            D=2:  p ≤ 1/2  (threshold shifted from 1/4 under old SWAP map)
            D>2:  p ≤ D/(2√(D−1)+D)  (slightly below 1/2 for finite D)
        """
        p = np.asarray(p, dtype=float)
        F0 = np.full_like(p, 1.0 / D)   # default: completely mixed state (F=1/D)

        valid = p < 1.0
        p_v = p[valid]

        disc = 1.0 - 4.0 * (D - 1) * p_v**2 / (D**2 * (1.0 - p_v)**2)
        good = disc >= 0.0

        idx = np.where(valid)[0][good]
        F0[idx] = 0.5 * (1.0 + np.sqrt(disc[good]))

        return F0

    # ------------------------------------------------------------------
    # Comprehensive 5-panel figure
    # ------------------------------------------------------------------

    def plot_comprehensive_grid_centered_gridspec(self, save_format: str = "pdf") -> str:
        """
        5-panel figure on a 3×6 GridSpec, all panels using the VD map.

        Layout:
            Row 0:  (a) F vs t  [p=0.1, ℓ=0..3]    (b) F vs t  [ℓ=1, p=0.1..0.3]
            Row 1:  (c) F₀ vs p [ℓ=0..3]            (d) γ_L vs p [ℓ=0..3,10,20]
            Row 2:            (e) γ_L vs ℓ [p=0.1..0.7]  (centered)

        Physics used throughout (VD map, D=2):
          - Noise:         r → (1-p) r
          - Purification:  r → P(r) = 2r/(1+r²)   [VD, replaces 4r/(3+r²)]
          - Fidelity:      F = (1+r)/2
          - F₀ (ℓ=1):     r_fix = √((1-2p)/(1-p)²),  threshold p=1/2
          - F₀ (ℓ=2):     r_fix = √((2√(1+p²)−(1+2p))/(1-p)²),  threshold p=3/4
          - γ_L:          F(t=0) - F(t=1)  starting from r₀=1
        """
        import matplotlib.gridspec as gridspec

        # Shared font sizes — kept consistent across all panels.
        # Label fontsize is 40 (not 50) so right-column ylabels don't bleed left.
        FS_LABEL = 40
        FS_TICK  = 30
        FS_LEGEND = 20

        fig = plt.figure(figsize=(20, 18))
        
        # Create 3×6 grid for precise positioning control
        gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.9)
        
        # Position plots for equal sizes and visual balance
        ax1 = fig.add_subplot(gs[0, 1:3])   # Row 1, columns 1-2 (Plot a)
        ax2 = fig.add_subplot(gs[0, 3:5])   # Row 1, columns 3-4 (Plot b)  
        ax3 = fig.add_subplot(gs[1, 1:3])   # Row 2, columns 1-2 (Plot c)
        ax4 = fig.add_subplot(gs[1, 3:5])   # Row 2, columns 3-4 (Plot d)
        ax5 = fig.add_subplot(gs[2, 2:4])   # Row 3, columns 2-3 (Plot e - centered)

        colors = ["red", "green", "blue", "orange", "purple",
                  "brown", "pink", "gray", "olive", "cyan"]

        # ── (a) Fidelity vs cycles, p=0.1, ℓ=0..3 ────────────────────────────
        ax   = ax1
        p_fixed = 0.1
        n_iter  = 20
        r0      = 1.0

        for i, ell in enumerate([0, 1, 2, 3]):
            traj_r = self.iterate_r(r0=r0, p=p_fixed, ell=ell, n_iter=n_iter)
            traj_F = (1 + traj_r) / 2
            it     = np.arange(len(traj_F))
            label  = 'No QEC' if ell == 0 else rf'$\ell={ell}$'
            style  = ':' if ell == 0 else '-'
            ax.plot(it, traj_F,
                    marker=_mk(i), color=colors[i], linestyle=style,
                    label=label, markevery=max(1, len(it)//8),
                    markersize=12, linewidth=2)

        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=FS_LABEL)
        ax.set_ylabel('Fidelity, F', fontsize=FS_LABEL)
        ax.set_xlim(0, n_iter)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=FS_LEGEND, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.99, 'a', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # ── (b) Fidelity vs cycles, ℓ=1, p=0.1..0.3 ──────────────────────────
        ax = ax2

        for i, p in enumerate([0.1, 0.2, 0.3]):
            traj_r = self.iterate_r(r0=r0, p=p, ell=1, n_iter=n_iter)
            traj_F = (1 + traj_r) / 2
            it     = np.arange(len(traj_F))
            ax.plot(it, traj_F,
                    marker=_mk(i), color=colors[i],
                    label=rf'$p={p}$',
                    markevery=max(1, len(it)//8),
                    markersize=12, linewidth=2)

        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=FS_LABEL)
        ax.set_ylabel(r'Fidelity, $F$', fontsize=FS_LABEL)
        ax.set_xlim(0, n_iter)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=FS_LEGEND, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.98, 'b', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # ── (c) Steady-state F₀ vs p, ℓ=0..3 ─────────────────────────────────
        # Extended to p=1.0 so thresholds (ℓ=1→p=0.5, ℓ=2→p=0.75, ℓ=3→p→∞)
        # are all visible.  p=1 is excluded (singular) so we stop at 0.995.
        ax = ax3
        p_range = np.linspace(0.0, 0.995, 400)

        for i, ell in enumerate([0, 1, 2, 3]):
            if ell == 0:
                F0_vals = np.full_like(p_range, 0.5)
                ax.plot(p_range, F0_vals,
                        color=colors[i], linewidth=3,
                        marker=_mk(i), markevery=50, markersize=10,
                        label='No QEC', linestyle='dotted')
            else:
                r_fix = self.rfix_general(p_range, ell)
                F_fix = (1 + r_fix) / 2
                ax.plot(p_range, F_fix,
                        color=colors[i], linewidth=3,
                        marker=_mk(i), markevery=50, markersize=10,
                        label=rf'$\ell={ell}$')

        ax.set_xlabel('Physical Error Rate, p', fontsize=FS_LABEL)
        ax.set_ylabel(r'$F_{0}$', fontsize=FS_LABEL)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=FS_LEGEND, loc='best', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.98, 'c', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # ── (d) Logical error γ_L vs p, various ℓ ─────────────────────────────
        ax = ax4
        p_range_gamma = np.linspace(0.01, 1.0, 50)
        r0_gamma = 1.0

        for i, ell in enumerate([0, 1, 2, 3, 10, 20]):
            gamma_vals = np.array([
                self.calculate_gamma_first_drop(r0_gamma, pv, ell)
                for pv in p_range_gamma
            ])
            label = 'No QEC' if ell == 0 else rf'$\ell={ell}$'
            style = ':' if ell == 0 else '-'
            ax.plot(p_range_gamma, gamma_vals,
                    color=colors[i], linewidth=3,
                    marker=_mk(i), markevery=5, markersize=12,
                    label=label, linestyle=style)

        ax.set_xlabel('Physical Error Rate, p', fontsize=FS_LABEL)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=FS_LABEL)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
        ax.set_ylim(-0.05, 0.6)
        ax.legend(fontsize=FS_LEGEND, loc='upper left', ncol=1, frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.98, 'd', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # ── (e) Logical error γ_L vs ℓ, various p ─────────────────────────────
        ax = ax5
        ell_range   = np.arange(0, 21)
        gamma_colors = ["green", "blue", "orange", "red"]

        for i, p in enumerate([0.1, 0.3, 0.5, 0.7]):
            gamma_vals = np.array([
                self.calculate_gamma_first_drop(r0_gamma, p, ell)
                for ell in ell_range
            ])
            ax.plot(ell_range, gamma_vals,
                    marker=_mk(i), color=gamma_colors[i], linewidth=3,
                    markersize=12, markevery=2,
                    label=rf'$p={p}$')

        ax.set_xlabel(r'Purification Rounds, $\ell$', fontsize=FS_LABEL)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=FS_LABEL)
        ax.set_xlim(0, 5)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_ylim(1e-7, 1.0)
        ax.set_yscale('log')
        ax.legend(fontsize=FS_LEGEND, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.97, 'e', transform=ax.transAxes, fontsize=32,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        filename = f"comprehensive_grid_centered_gridspec.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_comprehensive_grid_2x2(self, save_format: str = "pdf") -> str:
        """
        4-panel figure on a 2×6 GridSpec, all panels using the VD map.

        Layout:
            Row 0:  (a) F vs t  [p=0.1, ℓ=0..3]    (b) F vs t  [ℓ=1, p=0.1..0.3]
            Row 1:  (c) F₀ vs p [ℓ=0..3]            (d) γ_L vs p [ℓ=0..3,10,20]

        Physics used throughout (VD map, D=2):
          - Noise:         r → (1-p) r
          - Purification:  r → P(r) = 2r/(1+r²)   [VD, replaces 4r/(3+r²)]
          - Fidelity:      F = (1+r)/2
          - F₀ (ℓ=1):     r_fix = √((1-2p)/(1-p)²),  threshold p=1/2
          - F₀ (ℓ=2):     r_fix = √((2√(1+p²)−(1+2p))/(1-p)²),  threshold p=3/4
          - γ_L:          F(t=0) - F(t=1)  starting from r₀=1
        """
        import matplotlib.gridspec as gridspec

        # Shared font sizes — kept consistent across all panels.
        # Label fontsize is 40 (not 50) so right-column ylabels don't bleed left.
        FS_LABEL = 40
        FS_TICK  = 30
        FS_LEGEND = 20

        fig = plt.figure(figsize=(20, 12))
        
        # Create 2×6 grid for precise positioning control
        gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.9)
        
        # Position plots for equal sizes and visual balance
        ax1 = fig.add_subplot(gs[0, 1:3])   # Row 1, columns 1-2 (Plot a)
        ax2 = fig.add_subplot(gs[0, 3:5])   # Row 1, columns 3-4 (Plot b)  
        ax3 = fig.add_subplot(gs[1, 1:3])   # Row 2, columns 1-2 (Plot c)
        ax4 = fig.add_subplot(gs[1, 3:5])   # Row 2, columns 3-4 (Plot d)

        colors = ["red", "green", "blue", "orange", "purple",
                  "brown", "pink", "gray", "olive", "cyan"]

        # ── (a) Fidelity vs cycles, p=0.1, ℓ=0..3 ────────────────────────────
        ax   = ax1
        p_fixed = 0.1
        n_iter  = 20
        r0      = 1.0

        for i, ell in enumerate([0, 1, 2, 3]):
            traj_r = self.iterate_r(r0=r0, p=p_fixed, ell=ell, n_iter=n_iter)
            traj_F = (1 + traj_r) / 2
            it     = np.arange(len(traj_F))
            label  = 'No QEC' if ell == 0 else rf'$\ell={ell}$'
            style  = ':' if ell == 0 else '-'
            ax.plot(it, traj_F,
                    marker=_mk(i), color=colors[i], linestyle=style,
                    label=label, markevery=max(1, len(it)//8),
                    markersize=12, linewidth=2)

        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=FS_LABEL)
        ax.set_ylabel('Fidelity, F', fontsize=FS_LABEL)
        ax.set_xlim(0, n_iter)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=FS_LEGEND, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.99, 'a', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # ── (b) Fidelity vs cycles, ℓ=1, p=0.1..0.3 ──────────────────────────
        ax = ax2

        for i, p in enumerate([0.1, 0.2, 0.3]):
            traj_r = self.iterate_r(r0=r0, p=p, ell=1, n_iter=n_iter)
            traj_F = (1 + traj_r) / 2
            it     = np.arange(len(traj_F))
            ax.plot(it, traj_F,
                    marker=_mk(i), color=colors[i],
                    label=rf'$p={p}$',
                    markevery=max(1, len(it)//8),
                    markersize=12, linewidth=2)

        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=FS_LABEL)
        ax.set_ylabel(r'Fidelity, $F$', fontsize=FS_LABEL)
        ax.set_xlim(0, n_iter)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=FS_LEGEND, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.98, 'b', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # ── (c) Steady-state F₀ vs p, ℓ=0..3 ─────────────────────────────────
        # Extended to p=1.0 so thresholds (ℓ=1→p=0.5, ℓ=2→p=0.75, ℓ=3→p→∞)
        # are all visible.  p=1 is excluded (singular) so we stop at 0.995.
        ax = ax3
        p_range = np.linspace(0.0, 0.995, 400)

        for i, ell in enumerate([0, 1, 2, 3]):
            if ell == 0:
                F0_vals = np.full_like(p_range, 0.5)
                ax.plot(p_range, F0_vals,
                        color=colors[i], linewidth=3,
                        marker=_mk(i), markevery=50, markersize=10,
                        label='No QEC', linestyle='dotted')
            else:
                r_fix = self.rfix_general(p_range, ell)
                F_fix = (1 + r_fix) / 2
                ax.plot(p_range, F_fix,
                        color=colors[i], linewidth=3,
                        marker=_mk(i), markevery=50, markersize=10,
                        label=rf'$\ell={ell}$')

        ax.set_xlabel('Physical Error Rate, p', fontsize=FS_LABEL)
        ax.set_ylabel(r'$F_{0}$', fontsize=FS_LABEL)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=FS_LEGEND, loc='best', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.98, 'c', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # ── (d) Logical error γ_L vs p, various ℓ ─────────────────────────────
        ax = ax4
        p_range_gamma = np.linspace(0.01, 1.0, 50)
        r0_gamma = 1.0

        for i, ell in enumerate([0, 1, 2, 3, 10, 20]):
            gamma_vals = np.array([
                self.calculate_gamma_first_drop(r0_gamma, pv, ell)
                for pv in p_range_gamma
            ])
            label = 'No QEC' if ell == 0 else rf'$\ell={ell}$'
            style = ':' if ell == 0 else '-'
            ax.plot(p_range_gamma, gamma_vals,
                    color=colors[i], linewidth=3,
                    marker=_mk(i), markevery=5, markersize=12,
                    label=label, linestyle=style)

        ax.set_xlabel('Physical Error Rate, p', fontsize=FS_LABEL)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=FS_LABEL)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
        ax.set_ylim(-0.05, 0.6)
        ax.legend(fontsize=FS_LEGEND, loc='upper left', ncol=1, frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
        ax.text(0.97, 0.98, 'd', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        filename = f"comprehensive_2x2_grid.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Additional single panels
    # ------------------------------------------------------------------

    def plot_fout_vs_f_gamma_system(self, save_format: str = "pdf") -> str:
        """F_out vs F for the GHZ/gamma family."""
        F = np.linspace(0.0, 1.0, 500)
        gamma = 2 * F - 1
        valid = (gamma >= 0) & (gamma <= 1)
        F_v = F[valid]; gv = gamma[valid]
        gamma_prime = 4 * gv / (3 + gv**2)
        F_out = (1 + gamma_prime) / 2

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(F_v, F_out, marker='o', color='blue', linewidth=3,
                markevery=25, markersize=12)
        ax.plot(F, F, '--', color='gray', linewidth=2, alpha=0.7, label='Identity')

        ax.set_xlabel(r'Input Fidelity, $F$', fontsize=40)
        ax.set_ylabel(r'Output Fidelity, $F_{\mathrm{out}}$', fontsize=40)
        ax.set_xlim(0.5, 1); ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=18, frameon=False)
        plt.tight_layout()

        filename = f"fidelity_evolution_gamma_system.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    def plot_gamma_vs_purification_level(self, save_format: str = "pdf") -> str:
        """Logical error γ vs purification level ℓ for several p."""
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["red", "green", "blue", "orange"]
        r0 = 1.0

        for i, p in enumerate([0.1, 0.3, 0.5, 0.7]):
            gamma_vals = np.array([
                self.calculate_gamma_first_drop(r0, p, ell)
                for ell in range(21)
            ])
            ax.plot(np.arange(21), gamma_vals,
                    marker=_mk(i), color=colors[i], linewidth=3,
                    markersize=12, markevery=2, label=rf'$p={p}$')

        ax.set_xlabel(r'Purification Level, $\ell$', fontsize=40)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=40)
        ax.set_xlim(0, 20); ax.set_ylim(1e-7, 1.0)
        ax.set_yscale('log')
        ax.legend(fontsize=24, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=32)
        ax.text(0.97, 0.98, 'e', transform=ax.transAxes, fontsize=36,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        plt.tight_layout()

        filename = f"gamma_vs_purification_level.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def generate_all_requested(self, save_format: str = "pdf") -> Dict[str, Optional[str]]:
        print("\n" + "="*70)
        print("GENERATING ANALYTIC THEORY FIGURES  (VD map)")
        print("="*70)

        out: Dict[str, Optional[str]] = {}

        print("\n1) F_out vs F (isotropic/Werner family, VD map)...")
        out['fout_vs_f'] = self.plot_fout_vs_f_isotropic(save_format=save_format)

        print("\n2) Comprehensive 5-panel grid (VD-consistent)...")
        out['comprehensive_5panel'] = self.plot_comprehensive_grid_centered_gridspec(
            save_format=save_format
        )
        out['comprehensive_4panel'] = self.plot_comprehensive_grid_2x2(
            save_format=save_format
        )

        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(f"Figures saved to: {self.figures_dir}")
        return out


def main():
    import sys
    figures_dir = "figures/theory_ana_plots"
    save_format = "pdf"

    if '--figures-dir' in sys.argv:
        idx = sys.argv.index('--figures-dir')
        if idx + 1 < len(sys.argv):
            figures_dir = sys.argv[idx + 1]
    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        if idx + 1 < len(sys.argv):
            save_format = sys.argv[idx + 1]

    plotter = AnalyticTheoryPlotter(figures_dir)
    return plotter.generate_all_requested(save_format=save_format)


if __name__ == "__main__":
    main()