"""
Analytic theory plots for SWAP-based purification (no CSVs required).
Generates figures:
  (1) F_out vs F (isotropic family)
  (2) Error evolution with general bounds (isotropic family)
  (5) GHZ per-round error ratio vs coherence gamma
  (7) Dephasing anisotropy vs isotropy (single qubit, equatorial target)
  (8) Round-count scaling n_* vs M for GHZ (with/without isotropization)
  (9) F_out vs dimension D at fixed F

Saves PDFs under: figures/theory_ana_plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# ---------------------------------------------------------------------
# Publication-quality plotting (match your sample setup)
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

# Markers (same approach as your sample)
MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'h', '*']
def _mk(i: int) -> str:
    """Distinct marker for series index i (wraps automatically)."""
    return MARKERS[i % len(MARKERS)]

# ---------------------------------------------------------------------
# Analytic helpers (equations from the text/appendix)
# ---------------------------------------------------------------------

def T_isotropic(F: float, D: int) -> float:
    """Register purity for the isotropic/commuting family."""
    return F**2 + (1 - F)**2 / (D - 1)

def Fout_isotropic(F: np.ndarray, D: int) -> np.ndarray:
    """Exact F_out(F, D) for isotropic/commuting family."""
    denom = 1 + F**2 + (1 - F)**2 / (D - 1)
    return (F + F**2) / denom

def Fout_isotropic_v2(F: np.ndarray, D: int) -> np.ndarray:
    """Exact F_out(F, D) for isotropic/commuting family for the updated equation model."""
    numerator = F * (1+F) * (D-1)
    denominator = D-2*F+D*(F**2)
    return numerator / denominator

def err_bounds(F: float, D: int) -> Tuple[float, float]:
    """
    Lower/upper bounds on next-round error given current F and T (isotropic case),
    derived from F^2 <= <psi|rho^2|psi> <= F.
    Returns (lower, upper).
    """
    T = T_isotropic(F, D)
    lower = (1 + T - 2*F) / (1 + T)
    upper = (1 + T - F - F**2) / (1 + T)
    return lower, upper

def bloch_update_radius(r: float) -> float:
    """
    Qubit Bloch-vector radial update under SWAP purification:
        r -> 4 r / (3 + r^2)
    """
    return (4.0 * r) / (3.0 + r*r)

def ghz_gamma_update(gamma: float) -> float:
    """
    GHZ/CAT coherence update in its two-dimensional block:
        gamma -> 4 gamma / (3 + gamma^2)
    """
    return (4.0 * gamma) / (3.0 + gamma*gamma)

def ghz_err_ratio_per_round(gamma: np.ndarray) -> np.ndarray:
    """
    Exact per-round error ratio for GHZ/CAT:
        eps_out / eps = (3 - gamma) / (3 + gamma^2)
    """
    return (3.0 - gamma) / (3.0 + gamma*gamma)

def alpha_from_betaz(beta_z: float) -> float:
    """Isotropized contraction factor alpha = (1 + 2 beta_z) / 3."""
    return (1.0 + 2.0 * beta_z) / 3.0

def n_star_from_gamma0(gamma0: float) -> float:
    """Rounds n_* to lift small gamma0 to a constant (M-independent) target ~ O(1):
       n_* ~ log(1/gamma0) / log(4/3) in the small-gamma linear regime."""
    return np.log(1.0 / gamma0) / np.log(4.0 / 3.0)

# ---------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------

class AnalyticTheoryPlotter:
    def __init__(self, figures_dir: str = "figures/theory_ana_plots"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # (1) F_out vs F, isotropic family (use viridis shades per D)
    def plot_fout_vs_f_isotropic(self,
                                 D_list: Optional[List[int]] = None,
                                 save_format: str = "pdf") -> str:
        if D_list is None:
            D_list = [2, 4, 32]
        F = np.linspace(0.0, 1.0, 500)

        fig, ax = plt.subplots(figsize=(10, 8))
        # colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(D_list)))
        # colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6']  # Red, Green, Blue, Orange, Purple
        colors = ["red", "green", "blue", "orange", "purple"]

        for i, D in enumerate(D_list):
            Fout = Fout_isotropic_v2(F, D)
            ax.plot(F, Fout, marker=_mk(i), color=colors[i], label=f'D={D}', markevery=50, markersize=12)
        ax.plot(F, F, '--', color='gray', linewidth=2, alpha=0.7, label='Identity')

        ax.set_xlabel(r'Input Fidelity, $F$', fontsize=40)
        ax.set_ylabel(r"Output Fidelity, $F'$", fontsize=40)
        # ax.set_title(r'Fidelity Evolution (Isotropic Family)', fontsize=40)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=22, frameon=False)
        plt.tight_layout()

        filename = f"fout_vs_f_isotropic.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)
    

    # (2) Error evolution with bounds (isotropic recursion) — optional viridis touch
    def plot_error_evolution_with_bounds(self,
                                         F0: float = 0.7,
                                         D: int = 8,
                                         n_steps: int = 12,
                                         save_format: str = "pdf") -> str:
        # exact recursion
        F_exact = [F0]
        for _ in range(n_steps):
            F_exact.append(Fout_isotropic(np.array([F_exact[-1]]), D)[0])
        F_exact = np.array(F_exact)
        eps_exact = 1.0 - F_exact

        # bounds at each step (computed from previous F)
        eps_lower, eps_upper = [], []
        for F in F_exact[:-1]:
            low, up = err_bounds(F, D)
            eps_lower.append(low); eps_upper.append(up)
        eps_lower = np.array(eps_lower)
        eps_upper = np.array(eps_upper)

        fig, ax = plt.subplots(figsize=(10, 8))
        n = np.arange(len(eps_exact))
        v = plt.cm.viridis([0.15, 0.55, 0.85])  # a few viridis shades
        exact_color = v[2]   # darker
        bound_color = v[0]   # lighter

        ax.semilogy(n, eps_exact, '-', marker=_mk(0), color=exact_color,
                    label='Exact error', linewidth=3)
        ax.semilogy(n[1:], eps_lower, '--', marker=_mk(1), color=bound_color,
                    label='Lower bound', alpha=0.95)
        ax.semilogy(n[1:], eps_upper, '--', marker=_mk(2), color=bound_color,
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

    # (5) GHZ per-round error ratio vs gamma — single curve (color can be any; use viridis mid)
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
    #     Use SAME viridis color for the pair (no twirl vs isotropized)
    def plot_anisotropy_vs_isotropy_single_qubit(self,
                                                 beta_z: float = 0.5,
                                                 n_steps: int = 12,
                                                 save_format: str = "pdf") -> str:
        # No twirl: start r0 = beta_z
        r_no = [beta_z]
        for _ in range(n_steps):
            r_no.append(bloch_update_radius(r_no[-1]))
        r_no = np.array(r_no)

        # Isotropized: r0 = alpha = (1 + 2 beta_z)/3
        alpha = alpha_from_betaz(beta_z)
        r_iso = [alpha]
        for _ in range(n_steps):
            r_iso.append(bloch_update_radius(r_iso[-1]))
        r_iso = np.array(r_iso)

        F_no = 0.5 * (1.0 + r_no)
        F_iso = 0.5 * (1.0 + r_iso)

        fig, ax = plt.subplots(figsize=(10, 8))
        n = np.arange(len(F_no))
        base_color = plt.cm.viridis(0.7)  # pick a viridis shade

        ax.plot(n, F_no, '-', marker=_mk(0), color=base_color,
                label=rf'No twirl ($\beta_z={beta_z:.2f}$)')
        ax.plot(n, F_iso, '--', marker=_mk(1), color=base_color,
                label=fr'Isotropized ($\alpha=\frac{{1+2\beta_z}}{{3}}={alpha:.2f}$)')

        ax.set_xlabel(r'Purification Level, $n$', fontsize=25)
        ax.set_ylabel(r'Fidelity, $F_n$', fontsize=25)
        ax.set_title(r'Dephasing Anisotropy vs Isotropy (Single Qubit, Equator)', fontsize=30)
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()

        filename = f"anisotropy_vs_isotropy_qubit_betaz_{beta_z:.2f}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)

    # (8) Round-count scaling n_* vs M for GHZ — SAME viridis color per beta (solid vs dashed)
    def plot_round_count_vs_M(self,
                              beta_z_list: Optional[List[float]] = None,
                              M_list: Optional[List[int]] = None,
                              save_format: str = "pdf") -> str:
        if beta_z_list is None:
            beta_z_list = [0.6, 0.8, 0.9]
        if M_list is None:
            M_list = list(range(2, 65, 2))

        fig, ax = plt.subplots(figsize=(10, 8))
        # Distinct viridis shades for each beta
        v_positions = np.linspace(0.0, 1.0, len(beta_z_list))
        v_colors = [plt.cm.viridis(pos) for pos in v_positions]

        for i, beta_z in enumerate(beta_z_list):
            alpha = alpha_from_betaz(beta_z)
            color_i = v_colors[i]  # SAME color for the pair

            n_no, n_iso = [], []
            for M in M_list:
                gamma0_no = beta_z**M
                gamma0_iso = alpha**M
                n_no.append(n_star_from_gamma0(gamma0_no))
                n_iso.append(n_star_from_gamma0(gamma0_iso))

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

    # (9) F_out vs D at fixed F — single series (use any viridis shade)
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
        ax.set_title(r'Isotropic Family: $F_{\mathrm{out}}$ vs $D$ at Fixed $F$', fontsize=30)
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()

        filename = f"fout_vs_D_fixedF_{F0:.2f}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved {filename}")
        return str(filepath)
    
    
    
    def plot_fout_vs_f_gamma_system(self,
                                save_format: str = "pdf") -> str:
        """
        Plot fidelity evolution for gamma-based system.
        
        Uses the equations:
        - γ ∈ [0,1]
        - γ' = 4γ/(3+γ²)  
        - F = (1+γ)/2
        - F_out = (1+γ')/2
        
        Creates a single plot showing output fidelity vs input fidelity.
        """
        
        # Create F range - since γ ∈ [0,1] and F = (1+γ)/2, valid F ∈ [0.5, 1]
        # But keeping full range [0, 1] for consistency with previous plots
        F = np.linspace(0.0, 1.0, 500)
        
        # Convert F to γ: F = (1+γ)/2 => γ = 2F - 1
        gamma = 2 * F - 1
        
        # Only use values where γ ∈ [0, 1] (i.e., F ∈ [0.5, 1])
        valid_mask = (gamma >= 0) & (gamma <= 1)
        F_valid = F[valid_mask]
        gamma_valid = gamma[valid_mask]
        
        # Apply transformations
        gamma_prime = 4 * gamma_valid / (3 + gamma_valid**2)
        F_out = (1 + gamma_prime) / 2

        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Fidelity evolution plot
        ax.plot(F_valid, F_out, marker='o', color='blue', linewidth=3, markevery=25, markersize=12)
        ax.plot(F, F, '--', color='gray', linewidth=2, alpha=0.7, label='Identity')

        ax.set_xlabel(r'Input Fidelity, $F$', fontsize=40)
        ax.set_ylabel(r'Output Fidelity, $F_{\mathrm{out}}$', fontsize=40)
        # ax.set_title(r'Fidelity Evolution (GHZ Family)', fontsize=30)
        ax.set_xlim(0.5, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=18, frameon=False)

        plt.tight_layout()

        filename = f"fidelity_evolution_gamma_system.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)
    
    
    ##############################################################
    ########
    
    def P(self, x: np.ndarray) -> np.ndarray:
        """One PQEC radial update: x -> 4x/(3+x^2)."""
        return 4.0 * x / (3.0 + x**2)
    
    def iterate_r(self, r0: float, p: float, ell: int, n_iter: int) -> np.ndarray:
        """
        Iterate r_{n+1} = P^{(ell)}((1-p) r_n), for n_iter steps.
        """
        a = 1.0 - p
        r = float(r0)
        traj = [r]
        for _ in range(n_iter):
            x = a * r              # noise once per iteration
            for _ in range(ell):   # ell PQEC layers
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
                ax.plot(
                    it,
                    traj,
                    marker=_mk(i),
                    color=colors[i % len(colors)],
                    label=fr"$r_0={r0}$",
                    markevery=max(1, len(it) // 10),
                    markersize=12,
                    linewidth=2,
                )

            ax.set_title(fr"$\ell={ell},\; p={p}$", fontsize=40)
            ax.set_xlabel("Cycle", fontsize=40)
            ax.set_ylabel(r"Bloch radius, $r=|\vec r|$", fontsize=40)
            ax.set_xlim(0, n_iter)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([0, n_iter // 4, n_iter // 2, 3 * n_iter // 4, n_iter])
            ax.tick_params(axis="both", which="major", labelsize=26, length=8, width=2)

            # Legend inside the plot
            ax.legend(
                loc="lower right",   # good default for these monotone curves
                fontsize=16,
                frameon=True,
                framealpha=0.9,
            )

            plt.tight_layout()
            filename = f"r_vs_iteration_p{p}_ell{ell}_multi_r0.{save_format}"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved {filename}")
        return str(filepath)
            
    def rfix_ell1(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        num = 1.0 - 4.0*p
        den = (1.0 - p)**2
        r2 = np.where(num > 0.0, num/den, 0.0)
        return np.sqrt(r2)

    def rfix_ell2(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        u = (4.0*np.sqrt(4.0*p**2 + 9.0) - 8.0*p - 9.0)/3.0
        r2 = np.where(u > 0.0, u/(1.0 - p)**2, 0.0)
        return np.sqrt(r2)

    def rfix_general(self, p: np.ndarray, ell: int) -> np.ndarray:
        """
        General fixed point calculation for any ell using iterative approach.
        For ell=0 (no purification), r_fix = (1-p)*r_init, so fix point is 0.
        For ell>=1, solve iteratively.
        """
        p = np.asarray(p, dtype=float)
        
        if ell == 0:
            # No purification: r -> (1-p)*r, so r_fix = 0
            return np.zeros_like(p)
        elif ell == 1:
            return self.rfix_ell1(p)
        elif ell == 2:
            return self.rfix_ell2(p)
        else:
            # For higher ell, use iterative approach
            r_fix = np.zeros_like(p)
            for i, p_val in enumerate(p):
                if p_val >= 1.0:
                    r_fix[i] = 0.0
                    continue
                
                # Binary search for fixed point
                r_low, r_high = 0.0, 1.0
                for _ in range(50):  # iterations for convergence
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
        """
        Calculate gamma = F(t=0) - F(t=1) where F = (1+r)/2.
        Returns the fidelity drop after first iteration.
        """
        traj = self.iterate_r(r0, p, ell, 1)
        r_initial = traj[0]
        r_after_one = traj[1]
        
        F_initial = (1 + r_initial) / 2
        F_after_one = (1 + r_after_one) / 2
        
        gamma = F_initial - F_after_one
        return max(0.0, gamma)  # Ensure non-negative

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

        ax.set_xlim(0, p_max)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"Physical Error Rate, $p$", fontsize=40)
        ax.set_ylabel(r"$r_{\mathrm{fix}}$", fontsize=40)

        # Bigger tick numbers (this is the main knob you asked for)
        ax.tick_params(axis="both", which="major", labelsize=24, length=8, width=2)
        # ax.tick_params(axis="both", which="minor", labelsize=24, length=4, width=1.5)

        # Legend inside the axes
        ax.legend(loc="lower left", fontsize=18, frameon=True)

        plt.tight_layout()
        out = self.figures_dir / f"rfix_vs_p.{save_format}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")
        return str(out)
    
    
    
    def F0_analytical(self, p: np.ndarray, D: int = 2) -> np.ndarray:
        """
        Calculate F₀ using the analytical formula:
        F₀ = (1/2)(1 + sqrt((D-2)² p² - 2pD(3D-2) + D²) / (D(1-p)))
        
        For D=2, this simplifies to:
        F₀ = (1/2)(1 + sqrt(1 - 4p) / (1-p))
        
        Valid only when 1-4p >= 0, i.e., p <= 0.25
        """
        p = np.asarray(p, dtype=float)
        
        if D == 2:
            # Simplified formula for D=2
            valid_mask = (p <= 0.25) & (p < 1.0) & (p >= 0.0)
            F0 = np.zeros_like(p)
            
            p_valid = p[valid_mask]
            if len(p_valid) > 0:
                sqrt_term = np.sqrt(1 - 4*p_valid)
                F0[valid_mask] = 0.5 * (1 + sqrt_term / (1 - p_valid))
            
            # For p > 0.25 or p >= 1, F₀ = 0.5 (completely mixed)
            invalid_mask = ~valid_mask & (p < 1.0)
            F0[invalid_mask] = 0.5
            
            return F0
        else:
            # General formula for arbitrary D
            valid_mask = p < 1.0
            F0 = np.full_like(p, 0.5)  # Default to completely mixed state
            
            p_valid = p[valid_mask]
            if len(p_valid) > 0:
                D_term = (D-2)**2 * p_valid**2
                cross_term = -2 * p_valid * D * (3*D - 2)
                constant_term = D**2
                
                discriminant = D_term + cross_term + constant_term
                valid_discriminant = discriminant >= 0
                
                if np.any(valid_discriminant):
                    sqrt_term = np.sqrt(discriminant[valid_discriminant])
                    denom = D * (1 - p_valid[valid_discriminant])
                    
                    mask_idx = np.where(valid_mask)[0][valid_discriminant]
                    F0[mask_idx] = 0.5 * (1 + sqrt_term / denom)
            
            return F0


    def plot_comprehensive_2x2_grid(self, save_format: str = "pdf") -> str:
        """
        Create a 2x2 grid of plots:
        1. Top left: Fidelity vs iteration rounds for p=0.1 with ℓ = 0,1,2,3
        2. Top right: Fidelity vs iteration rounds for ℓ=1 with p=0.1,0.2,0.3
        3. Bottom left: r_fix vs p with ℓ=0,1,2,3
        4. Bottom right: gamma vs p with ℓ=0,1,2,3,10,20
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        
        # Top left: Fidelity vs iteration for p=0.1, different ell values
        ax = axes[0, 0]
        p_fixed = 0.1
        ell_list = [0, 1, 2, 3]
        r0 = 1.0  # Initial Bloch radius
        n_iter = 20
        
        for i, ell in enumerate(ell_list):
            traj_r = self.iterate_r(r0=r0, p=p_fixed, ell=ell, n_iter=n_iter)
            traj_F = (1 + traj_r) / 2  # Convert to fidelity
            iterations = np.arange(len(traj_F))
            if ell == 0:
                ax.plot(
                iterations, traj_F,
                marker=_mk(i), color=colors[i % len(colors)], linestyle=':',
                label=f'No QEC',
                markevery=max(1, len(iterations) // 8),
                markersize=12, linewidth=2
            )
            else:
                ax.plot(
                    iterations, traj_F,
                    marker=_mk(i), color=colors[i % len(colors)],
                    label=rf'$\ell={ell}$',
                    markevery=max(1, len(iterations) // 8),
                    markersize=12, linewidth=2
                )
        
        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=50)
        ax.set_ylabel('Fidelity, F', fontsize=50)
        # ax.set_title(f'Fidelity vs Iterations\n(p={p_fixed})', fontsize=40)
        ax.set_xlim(0, n_iter)
        ax.set_xticks(np.arange(0, n_iter + 1, 2))
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=22, loc='lower left', frameon=False)
        ax.tick_params(axis="y", which="major", labelsize=36)
        ax.tick_params(axis="x", which="major", labelsize=32)
        
        # Add subplot label (a)
        ax.text(0.97, 0.99, 'a', transform=ax.transAxes, fontsize=36, 
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # Top right: Fidelity vs iteration for ell=1, different p values
        ax = axes[0, 1]
        ell_fixed = 1
        p_list = [0.1, 0.2, 0.3]
        
        for i, p in enumerate(p_list):
            traj_r = self.iterate_r(r0=r0, p=p, ell=ell_fixed, n_iter=n_iter)
            traj_F = (1 + traj_r) / 2  # Convert to fidelity
            iterations = np.arange(len(traj_F))
            
            ax.plot(
                iterations, traj_F,
                marker=_mk(i), color=colors[i % len(colors)],
                label=rf'$p={p}$',
                markevery=max(1, len(iterations) // 8),
                markersize=12, linewidth=2
            )
        
        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=50)
        ax.set_ylabel(r'Fidelity, $F$', fontsize=50)
        # ax.set_title(f'Fidelity vs Iterations\n(ℓ={ell_fixed})', fontsize=20)
        ax.set_xlim(0, n_iter)
        ax.set_xticks(np.arange(0, n_iter + 1, 2))
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=22, loc='lower left', frameon=False)
        ax.tick_params(axis="y", which="major", labelsize=36)
        ax.tick_params(axis="x", which="major", labelsize=32)
        
        # Add subplot label (b)
        ax.text(0.97, 0.98, 'b', transform=ax.transAxes, fontsize=36, 
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # # Bottom left: r_fix vs p for different ell values
        # ax = axes[1, 0]
        # p_range = np.linspace(0.0, 0.6, 300)
        # ell_list_fix = [0, 1, 2, 3]
        
        # for i, ell in enumerate(ell_list_fix):
        #     r_fix = self.rfix_general(p_range, ell)
        #     if ell == 0:
        #         ax.plot(p_range, r_fix, 
        #            color=colors[i % len(colors)], linewidth=3,
        #            marker=_mk(i), markevery=30, markersize=12,
        #            label=f'No QEC', linestyle=':')
        #     else:   
        #         ax.plot(p_range, r_fix, 
        #             color=colors[i % len(colors)], linewidth=3, 
        #             marker=_mk(i), markevery=30, markersize=12,
        #             label=rf'$\ell={ell}$')
        
        # # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=40)
        # ax.set_xlabel('Physical Error Rate, p', fontsize=50)
        # ax.set_ylabel(r'$F_{0}$', fontsize=50)
        # # ax.set_title('Fixed Point vs Error Rate', fontsize=40)
        # ax.set_xlim(0, 0.6)
        # ax.set_ylim(-0.05, 1.05)
        # ax.legend(fontsize=22, loc='lower left', frameon=False)
        # ax.tick_params(axis="both", which="major", labelsize=36)
        
        # # Add subplot label (c)
        # ax.text(0.97, 0.98, 'c', transform=ax.transAxes, fontsize=36, 
        #         fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        
        # Bottom left: F₀ vs p using analytical formula for different ell values
        ax = axes[1, 0]
        p_range = np.linspace(0.0, 0.6, 300)
        ell_list_fix = [0, 1, 2, 3]
        
        for i, ell in enumerate(ell_list_fix):
            if ell == 0:
                F0_vals = np.full_like(p_range, 0.5)
                F0_vals[p_range == 0.0] = 1.0  # Perfect fidelity when p=0
                ax.plot(p_range, F0_vals, 
                    color=colors[i % len(colors)], linewidth=3,
                    marker=_mk(i), markevery=30, markersize=12,
                    label='No QEC', linestyle='dotted')
            else:
                # For ℓ > 0, convert r_fix to F_fix = (1 + r_fix)/2
                r_fix = self.rfix_general(p_range, ell)
                F_fix = (1 + r_fix) / 2
                ax.plot(p_range, F_fix, 
                        color=colors[i % len(colors)], linewidth=3, 
                        marker=_mk(i), markevery=30, markersize=12,
                        label=rf'$\ell={ell}$')
        
        ax.set_xlabel('Physical Error Rate, p', fontsize=50)
        ax.set_ylabel(r'$F_{0}$', fontsize=50)
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=22, loc='center left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=36)
        
        # Add subplot label (c)
        ax.text(0.97, 0.98, 'c', transform=ax.transAxes, fontsize=36, 
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # Bottom right: gamma vs p for different ell values
        ax = axes[1, 1]
        ell_list_gamma = [0, 1, 2, 3, 10, 20]
        p_range_gamma = np.linspace(0.01, 1.0, 50)  # Avoid p=0 for numerical stability
        r0_gamma = 1.0  # Initial condition for gamma calculation
        
        for i, ell in enumerate(ell_list_gamma):
            gamma_values = []
            for p_val in p_range_gamma:
                gamma = self.calculate_gamma_first_drop(r0_gamma, p_val, ell)
                gamma_values.append(gamma)
            
            gamma_values = np.array(gamma_values)
            if ell == 0:
                ax.plot(p_range_gamma, gamma_values,
                   color=colors[i % len(colors)], linewidth=3,
                   marker=_mk(i), markevery=5, markersize=12,
                   label=rf'No QEC', linestyle=':')
            else:
                ax.plot(p_range_gamma, gamma_values,
                    color=colors[i % len(colors)], linewidth=3,
                    marker=_mk(i), markevery=5, markersize=12,
                    label=rf'$\ell={ell}$')
        
        # ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=40)
        ax.set_xlabel('Physical Error Rate, p', fontsize=50)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=50)
        # ax.set_title('First Iteration Drop vs Error Rate', fontsize=20)
        ax.set_xlim(0.01, 1.0)
        ax.set_ylim(-0.05, 0.6)
        ax.legend(fontsize=22, loc='upper left', ncol=2, frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=36)
        
        # Add subplot label (d)
        ax.text(0.97, 0.98, 'd', transform=ax.transAxes, fontsize=36, 
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        plt.tight_layout()
        
        filename = f"comprehensive_2x2_grid.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)
    
    
    
    
    def plot_gamma_vs_purification_level(self, save_format: str = "pdf") -> str:
        """
        Plot logical error γ vs purification level ℓ for different physical error rates.
        
        Shows how increasing purification rounds reduces logical error across 
        different noise strengths. Complements the 2x2 grid by showing scaling behavior.
        
        X-axis: Purification level ℓ
        Y-axis: Logical error γ_L  
        Curves: Different p values (0.1, 0.3, 0.5, 0.7)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Parameters
        p_list = [0.1, 0.3, 0.5, 0.7]
        ell_range = np.arange(0, 21)  # ℓ = 0 to 20
        colors = ["red", "green", "blue", "orange"]  # Green to red for increasing p
        r0_initial = 1.0  # Initial Bloch radius
        
        for i, p in enumerate(p_list):
            gamma_values = []
            
            for ell in ell_range:
                gamma = self.calculate_gamma_first_drop(r0_initial, p, ell)
                gamma_values.append(gamma)
            
            gamma_values = np.array(gamma_values)
            
            # Plot with distinctive styling
            ax.plot(ell_range, gamma_values,
                    marker=_mk(i), color=colors[i], linewidth=3,
                    markersize=12, markevery=2,
                    label=rf'$p={p}$')
        
        # Formatting
        ax.set_xlabel(r'Purification Level, $\ell$', fontsize=40)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=40)
        ax.set_xlim(0, 20)
        ax.set_ylim(1e-7, 1.0)
        
        # Styling consistent with your other plots
        ax.legend(fontsize=24, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=32)
        # ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale to show wide range of gamma values more clearly
        
        ax.text(0.97, 0.98, 'e', transform=ax.transAxes, fontsize=36, 
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        plt.tight_layout()
        
        filename = f"gamma_vs_purification_level.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)


    def plot_comprehensive_grid_centered_gridspec(self, save_format: str = "pdf") -> str:
        """
        Complete GridSpec implementation for 5 equal-sized plots with perfect centering.
        
        Layout using 3 rows × 6 columns for precise control:
        ┌───┬───┬───┬───┬───┬───┐
        │   │ Plot a │   │ Plot b │   │ Row 1
        ├───┼───┼───┼───┼───┼───┤
        │   │ Plot c │   │ Plot d │   │ Row 2  
        ├───┼───┼───┼───┼───┼───┤
        │   │   │ Plot e │   │   │ Row 3 (centered)
        └───┴───┴───┴───┴───┴───┘
        
        All plots have identical dimensions and plot e is perfectly centered.
        """
        import matplotlib.gridspec as gridspec
        
        fig = plt.figure(figsize=(20, 18))
        
        # Create 3×6 grid for precise positioning control
        gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.9)
        
        # Position plots for equal sizes and visual balance
        ax1 = fig.add_subplot(gs[0, 1:3])   # Row 1, columns 1-2 (Plot a)
        ax2 = fig.add_subplot(gs[0, 3:5])   # Row 1, columns 3-4 (Plot b)  
        ax3 = fig.add_subplot(gs[1, 1:3])   # Row 2, columns 1-2 (Plot c)
        ax4 = fig.add_subplot(gs[1, 3:5])   # Row 2, columns 3-4 (Plot d)
        ax5 = fig.add_subplot(gs[2, 2:4])   # Row 3, columns 2-3 (Plot e - centered)
        
        colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        
        # ===== PLOT 1: Top left - Fidelity vs iteration (p=0.1, different ℓ) =====
        ax = ax1
        p_fixed = 0.1
        ell_list = [0, 1, 2, 3]
        r0 = 1.0
        n_iter = 20
        
        for i, ell in enumerate(ell_list):
            traj_r = self.iterate_r(r0=r0, p=p_fixed, ell=ell, n_iter=n_iter)
            traj_F = (1 + traj_r) / 2
            iterations = np.arange(len(traj_F))
            if ell == 0:
                ax.plot(iterations, traj_F,
                    marker=_mk(i), color=colors[i % len(colors)], linestyle=':',
                    label='No QEC', markevery=max(1, len(iterations) // 8),
                    markersize=10, linewidth=2)
            else:
                ax.plot(iterations, traj_F,
                    marker=_mk(i), color=colors[i % len(colors)],
                    label=rf'$\ell={ell}$', markevery=max(1, len(iterations) // 8),
                    markersize=12, linewidth=2)
        
        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=40)
        ax.set_ylabel('Fidelity, F', fontsize=40)
        ax.set_xlim(0, n_iter)
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=20, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=28)
        ax.text(0.97, 0.97, 'a', transform=ax.transAxes, fontsize=32, 
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # ===== PLOT 2: Top right - Fidelity vs iteration (ℓ=1, different p) =====
        ax = ax2
        ell_fixed = 1
        p_list = [0.1, 0.2, 0.3]
        
        for i, p in enumerate(p_list):
            traj_r = self.iterate_r(r0=r0, p=p, ell=ell_fixed, n_iter=n_iter)
            traj_F = (1 + traj_r) / 2
            iterations = np.arange(len(traj_F))
            ax.plot(iterations, traj_F,
                marker=_mk(i), color=colors[i % len(colors)],
                label=rf'$p={p}$', markevery=max(1, len(iterations) // 8),
                markersize=12, linewidth=2)
        
        ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=40)
        ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
        ax.set_xlim(0, n_iter)
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=20, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=28)
        ax.text(0.97, 0.97, 'b', transform=ax.transAxes, fontsize=32,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # ===== PLOT 3: Middle left - F₀ vs p (different ℓ) =====
        ax = ax3
        p_range = np.linspace(0.0, 0.6, 300)
        ell_list_fix = [0, 1, 2, 3]
        
        for i, ell in enumerate(ell_list_fix):
            if ell == 0:
                F0_vals = np.full_like(p_range, 0.5)
                F0_vals[p_range == 0.0] = 1.0
                ax.plot(p_range, F0_vals, 
                    color=colors[i % len(colors)], linewidth=3,
                    marker=_mk(i), markevery=30, markersize=10,
                    label='No QEC', linestyle=':')
            else:
                r_fix = self.rfix_general(p_range, ell)
                F_fix = (1 + r_fix) / 2
                ax.plot(p_range, F_fix, 
                    color=colors[i % len(colors)], linewidth=3, 
                    marker=_mk(i), markevery=30, markersize=12,
                    label=rf'$\ell={ell}$')
        
        ax.set_xlabel('Physical Error Rate, p', fontsize=40)
        ax.set_ylabel(r'$F_{0}$', fontsize=40)
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=20, loc='center left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=28)
        ax.text(0.97, 0.97, 'c', transform=ax.transAxes, fontsize=32,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # ===== PLOT 4: Middle right - gamma vs p (different ℓ) =====
        ax = ax4
        ell_list_gamma = [0, 1, 2, 3, 10, 20]
        p_range_gamma = np.linspace(0.01, 1.0, 50)
        r0_gamma = 1.0
        
        for i, ell in enumerate(ell_list_gamma):
            gamma_values = []
            for p_val in p_range_gamma:
                gamma = self.calculate_gamma_first_drop(r0_gamma, p_val, ell)
                gamma_values.append(gamma)
            
            gamma_values = np.array(gamma_values)
            if ell == 0:
                ax.plot(p_range_gamma, gamma_values,
                    color=colors[i % len(colors)], linewidth=3,
                    marker=_mk(i), markevery=5, markersize=12,
                    label='No QEC', linestyle=':')
            else:
                ax.plot(p_range_gamma, gamma_values,
                    color=colors[i % len(colors)], linewidth=3,
                    marker=_mk(i), markevery=5, markersize=12,
                    label=rf'$\ell={ell}$')
        
        ax.set_xlabel('Physical Error Rate, p', fontsize=40)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=40)
        ax.set_xlim(0.01, 1.0)
        ax.set_ylim(-0.05, 0.6)
        ax.legend(fontsize=20, loc='upper left', ncol=1, frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=28)
        ax.text(0.97, 0.97, 'd', transform=ax.transAxes, fontsize=32,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # ===== PLOT 5: Bottom center - gamma vs ℓ (different p) =====
        ax = ax5
        p_list_gamma = [0.1, 0.3, 0.5, 0.7]
        ell_range = np.arange(0, 21)
        gamma_colors = ["green", "blue", "orange", "red"]  # Green to red for increasing p
        
        for i, p in enumerate(p_list_gamma):
            gamma_values = []
            for ell in ell_range:
                gamma = self.calculate_gamma_first_drop(r0_gamma, p, ell)
                gamma_values.append(gamma)
            
            gamma_values = np.array(gamma_values)
            ax.plot(ell_range, gamma_values,
                marker=_mk(i), color=gamma_colors[i], linewidth=3,
                markersize=12, markevery=2,
                label=rf'$p={p}$')
        
        ax.set_xlabel(r'Purification Rounds, $\ell$', fontsize=40)
        ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=40)
        ax.set_xlim(0, 20)
        ax.set_ylim(1e-7, 1.0)
        ax.set_yscale('log')
        ax.legend(fontsize=20, loc='lower left', frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=28)
        ax.text(0.97, 0.97, 'e', transform=ax.transAxes, fontsize=32,
                fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
        
        # Save the figure
        filename = f"comprehensive_grid_centered_gridspec.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
        return str(filepath)



































        

    # Convenience driver
    def generate_all_requested(self, save_format: str = "pdf") -> Dict[str, Optional[str]]:
        print("\n" + "="*70)
        print("GENERATING ANALYTIC THEORY FIGURES")
        print("="*70)

        out: Dict[str, Optional[str]] = {}

        print("\n1) F_out vs F (isotropic family)...")
        out['fout_vs_f'] = self.plot_fout_vs_f_isotropic(save_format=save_format)

        ## print("\n2) Error evolution with bounds (isotropic family)...")
        ## out['err_evolution_bounds'] = self.plot_error_evolution_with_bounds(
        ##     F0=0.7, D=8, n_steps=12, save_format=save_format
        ## )

        ## print("\n5) GHZ per-round error ratio vs gamma...")
        ## out['ghz_err_ratio'] = self.plot_ghz_error_ratio(save_format=save_format)

        ## print("\n7) Dephasing anisotropy vs isotropy (single qubit)...")
        ## out['anisotropy_vs_iso'] = self.plot_anisotropy_vs_isotropy_single_qubit(
        ##     beta_z=0.5, n_steps=12, save_format=save_format
        ## )

        ## print("\n8) Round-count scaling n_* vs M (GHZ)...")
        ## out['round_count_vs_M'] = self.plot_round_count_vs_M(
        ##     beta_z_list=[0.6, 0.8, 0.9],
        ##     M_list=list(range(2, 65, 2)),
        ##     save_format=save_format
        ## )

        ## print("\n9) F_out vs D at fixed F...")
        ## out['fout_vs_D'] = self.plot_fout_vs_D_fixedF(F0=0.7, save_format=save_format)
        
        print("\n10. F_out vs F (GHZ system with gamma)...") 
        out['fout_vs_f_GHZ_system'] = self.plot_fout_vs_f_gamma_system(save_format=save_format)
        
        print("\n11. Bloch radius vs iteration under depolarizing + PQEC...")
        out['r_vs_iteration_multi_r0'] = self.plot_r_vs_iteration_multi_r0(
            p=0.1,
            ell_list=(1, 2),
            r0_list=(0.1, 0.5, 1.0),
            n_iter=20,
            save_format=save_format
        )
        print("\n12. Fixed-point Bloch radius vs depolarizing strength...")
        out['rfix_vs_p'] = self.plot_rfix_vs_p(
            figures_dir=self.figures_dir,
            p_max=0.5,
            n=500,
            save_format=save_format
        )
        
        print("\n13. Comprehensive 2x2 grid plot...")
        out['comprehensive_2x2_grid'] = self.plot_comprehensive_2x2_grid(save_format=save_format)
        out['comprehensive 5 plot grid'] = self.plot_comprehensive_grid_centered_gridspec(save_format=save_format)
        
        print("\n14. Gamma vs purification level...")
        out['gamma_vs_ell'] = self.plot_gamma_vs_purification_level(save_format=save_format)

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
    results = plotter.generate_all_requested(save_format=save_format)
    return results


if __name__ == "__main__":
    main()