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
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 8
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
    # def plot_fout_vs_f_isotropic(self,
    #                              D_list: Optional[List[int]] = None,
    #                              save_format: str = "pdf") -> str:
    #     if D_list is None:
    #         D_list = [2, 4, 8, 16, 32]
    #     F = np.linspace(0.0, 1.0, 500)

    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(D_list)))

    #     for i, D in enumerate(D_list):
    #         Fout = Fout_isotropic(F, D)
    #         ax.plot(F, Fout, marker='', color=colors[i], label=f'D={D}')
    #     ax.plot(F, F, '--', color='gray', linewidth=2, alpha=0.7, label='Identity')

    #     ax.set_xlabel(r'Input Fidelity, $F$', fontsize=25)
    #     ax.set_ylabel(r'Output Fidelity, $F_{\mathrm{out}}$', fontsize=25)
    #     ax.set_title(r'Fidelity Evolution (Isotropic Family)', fontsize=30)
    #     ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    #     ax.legend(loc='lower right', fontsize=14)
    #     plt.tight_layout()

    #     filename = f"fout_vs_f_isotropic.{save_format}"
    #     filepath = self.figures_dir / filename
    #     plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.close()
    #     print(f"Saved {filename}")
    #     return str(filepath)
    
    def plot_fout_vs_f_isotropic(self,
                                 D_list: Optional[List[int]] = None,
                                 save_format: str = "pdf") -> str:
        """
        Plot fidelity evolution and error reduction ratio for isotropic family.
        
        Creates two vertically aligned subplots:
        - Top: Output fidelity vs input fidelity 
        - Bottom: Error reduction ratio vs input fidelity
        
        Parameters:
        -----------
        D_list : List[int], optional
            List of D values to plot. Default: [2, 4, 8, 16, 32]
        save_format : str, optional
            File format for saving. Default: "pdf"
            
        Returns:
        --------
        str : Path to saved figure
        """
        if D_list is None:
            D_list = [2, 4, 8, 16, 32]
        F = np.linspace(0.0, 1.0, 500)

        # Create 2x1 subplot layout with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(D_list)))

        # Top subplot: Original fidelity plot
        for i, D in enumerate(D_list):
            Fout = Fout_isotropic(F, D)
            ax1.plot(F, Fout, marker='', color=colors[i], linewidth=3, label=f'D={D}')
        ax1.plot(F, F, '--', color='gray', linewidth=2, alpha=0.7, label='Identity')
        # ax1.axvline(x=0.5, linestyle='--', color='red', linewidth=2, alpha=0.8)

        ax1.set_ylabel(r'Output Fidelity, $F_{\mathrm{out}}$', fontsize=25)
        ax1.set_title(r'Fidelity Evolution (Isotropic Family)', fontsize=30)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='lower right', fontsize=14)
        # ax1.grid(True, alpha=0.3)

        # Bottom subplot: Error reduction ratio
        for i, D in enumerate(D_list):
            # Calculate error reduction ratio using the formula:
            # ε_out/ε = [1 - 2F + (1-F)²/(D-1)] / [(1-F)(1+F²+(1-F)²/(D-1))]
            
            F_calc = F[F <= 0.99999]  # Avoid numerical issues near F=1
            
            # Calculate components
            one_minus_F = 1 - F_calc
            term = one_minus_F**2 / (D - 1)
            
            # Numerator and denominator
            numerator = 1 - F_calc + term
            denominator = one_minus_F * (1 + F_calc**2 + term)
            
            # Calculate ratio with error handling
            with np.errstate(divide='ignore', invalid='ignore'):
                error_ratio = numerator / denominator
                
            # Filter valid values
            valid_mask = np.isfinite(error_ratio)
            ax2.plot(F_calc[valid_mask], error_ratio[valid_mask], 
                    marker='', color=colors[i], linewidth=3, label=f'D={D}')

        # Reference line at y=1 (no improvement)
        ax2.axhline(y=1, linestyle='--', color='gray', linewidth=2, alpha=0.7, label='No Improvement')
        # ax2.axvline(x=0.5, linestyle='--', color='red', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel(r'Input Fidelity, $F$', fontsize=25)
        ax2.set_ylabel(r'Error Reduction Ratio, $\frac{\varepsilon_{\mathrm{out}}}{\varepsilon}$', fontsize=25)
        ax2.set_title(r'Error Reduction Ratio (Isotropic Family)', fontsize=30)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.2)
        # ax2.legend(loc='upper right', fontsize=14)
        # ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"fout_vs_f_isotropic_combined.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
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
        Plot fidelity evolution and error reduction ratio for gamma-based system.
        
        Uses the equations:
        - γ ∈ [0,1]
        - γ' = 4γ/(3+γ²)  
        - F = (1+γ)/2
        - F_out = (1+γ')/2
        - Error reduction ratio = (3-γ)/(3+γ²)
        
        Creates two vertically aligned subplots:
        - Top: Output fidelity vs input fidelity 
        - Bottom: Error reduction ratio vs input fidelity
        Both plots include a vertical red dashed line at F = 0.5
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
        error_reduction_ratio = (3 - gamma_valid) / (3 + gamma_valid**2)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        # Top subplot: Fidelity evolution
        ax1.plot(F_valid, F_out, marker='', color='blue', linewidth=3)
        ax1.plot(F, F, '--', color='gray', linewidth=2, alpha=0.7, label='Identity')
        # ax1.axvline(x=0.5, linestyle='--', color='red', linewidth=2, alpha=0.8)

        ax1.set_ylabel(r'Output Fidelity, $F_{\mathrm{out}}$', fontsize=25)
        ax1.set_title(r'Fidelity Evolution (GHZ Family)', fontsize=30)
        ax1.set_xlim(0.5, 1)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='lower right', fontsize=14)
        # ax1.grid(True, alpha=0.3)

        # Bottom subplot: Error reduction ratio
        ax2.plot(F_valid, error_reduction_ratio, marker='', color='blue', linewidth=3)
        ax2.axhline(y=1, linestyle='--', color='gray', linewidth=2, alpha=0.7, label='No Improvement')
        # ax2.axvline(x=0.5, linestyle='--', color='red', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel(r'Input Fidelity, $F$', fontsize=25)
        ax2.set_ylabel(r'Error Reduction Ratio, $\frac{\varepsilon_{\mathrm{out}}}{\varepsilon}$', fontsize=25)
        ax2.set_title(r'Error Reduction Ratio (GHZ Family)', fontsize=30)
        ax2.set_xlim(0.5, 1)
        ax2.set_ylim(0, 1.2)
        ax2.legend(loc='upper left', fontsize=14)
        # ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"fout_vs_f_gamma_system.{save_format}"
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
