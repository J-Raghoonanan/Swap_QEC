"""
Figure generation for streaming QEC simulation data.
Loads CSV data from circuit simulations and creates publication-quality figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Publication-quality plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 40,
    'axes.labelsize': 40,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 20,
    'figure.titlesize': 40,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 12
})

COLORS = {
    'depolarizing': '#2E86AB',
    'dephasing': '#F18F01',
}

# Chosen for clear distinctness in print; extend if you often have many series.
MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'h', '*']

def _mk(i: int) -> str:
    """Return a distinct marker for series index i (wraps automatically)."""
    return MARKERS[i % len(MARKERS)]

class SimulationPlotter:
    """Generate figures from simulation CSV data."""
    
    def __init__(self, data_dir: str = "data/simulations_v2", 
                 figures_dir: str = "figures/results_v4_sim"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.depol_finals = self._load_csv('finals_circuit_depolarizing.csv')
        self.depol_steps = self._load_csv('steps_circuit_depolarizing.csv')
        
        # Load both dephasing datasets
        self.dephase_untwirled_finals = self._load_csv('finals_circuit_dephasing_v4_backup.csv')  # v4 for no twirling
        self.dephase_untwirled_steps = self._load_csv('steps_circuit_dephasing_v4_backup.csv')
        # self.dephase_untwirled_steps = self._load_csv('Backup_M1_dephasing_untwirled_v4_data.csv')
        self.dephase_twirled_finals = self._load_csv('finals_circuit_dephasing_v6.csv')  # v6 with proper twirling
        self.dephase_twirled_steps = self._load_csv('steps_circuit_dephasing_v6.csv')
        
        # Keep old references for backwards compatibility
        self.dephase_finals = self.dephase_twirled_finals  # Default to twirled
        self.dephase_steps = self.dephase_twirled_steps
        
        print(f"Loaded simulation data:")
        print(f"  Depolarizing finals: {len(self.depol_finals)} runs")
        print(f"  Depolarizing steps: {len(self.depol_steps)} merges")
        print(f"  Dephasing untwirled finals: {len(self.dephase_untwirled_finals)} runs")
        print(f"  Dephasing untwirled steps: {len(self.dephase_untwirled_steps)} merges")
        print(f"  Dephasing twirled finals: {len(self.dephase_twirled_finals)} runs")
        print(f"  Dephasing twirled steps: {len(self.dephase_twirled_steps)} merges")
        
        # ----------------------------
        # NEW: GlobalTwirl datasets
        # ----------------------------
        self.global_dir = Path("data/globalTwirl_simulations")
        self.global_dephase_steps = self._load_csv_from(self.global_dir, "steps_globalTwirl_dephase_z.csv")
        self.global_dephase_finals = self._load_csv_from(self.global_dir, "finals_globalTwirl_dephase_z.csv")
        
        self.subsetTwirl_dir = Path("data/subsetTwirling_simulations")
        self.subsetTwirl_dephase_steps = self._load_csv_from(self.subsetTwirl_dir, "steps_circuit_dephase_z__subset0.30.csv")
        self.subsetTwirl_dephase_finals = self._load_csv_from(self.subsetTwirl_dir, "finals_circuit_dephase_z__subset0.30.csv")
        
        self.subsetTwirl_dephase_steps_debug = self._load_csv_from(self.subsetTwirl_dir, "steps_circuit_dephase_z__subset0.30_DEBUG.csv")
        self.subsetTwirl_dephase_finals_debug = self._load_csv_from(self.subsetTwirl_dir, "finals_circuit_dephase_z__subset0.30_DEBUG.csv")

    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file if it exists."""
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
        
            # For steps files, extract N from run_id if not present
            if 'steps' in filename and 'N' not in df.columns and 'run_id' in df.columns:
                def extract_N(run_id):
                    # Format: M1_N512_dephase_z_iid_p_p0.50000_twirl
                    parts = run_id.split('_')
                    for part in parts:
                        if part.startswith('N'):
                            return int(part[1:])
                    return None
            
                df['N'] = df['run_id'].apply(extract_N)
        
            # Normalize twirling column name and convert to boolean
            if 'twirling_applied' in df.columns:
                # Convert string "TRUE"/"FALSE" to boolean and rename column
                df['twirling_enabled'] = df['twirling_applied'].map(
                    lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else False
                )
            elif 'twirling_enabled' in df.columns and df['twirling_enabled'].dtype == 'object':
                # Convert string to boolean if needed
                df['twirling_enabled'] = df['twirling_enabled'].map(
                    lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else x
                )
        
            return df
        else:
            print(f"Warning: {filename} not found")
            return pd.DataFrame()
        
    def _load_csv_from(self, base_dir: Path, filename: str) -> pd.DataFrame:
        """Load CSV file from an explicit directory if it exists."""
        filepath = base_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            return self._postprocess_df(df, filename_hint=filename)
        else:
            print(f"Warning: {filename} not found in {base_dir}")
            return pd.DataFrame()
        
    def _postprocess_df(self, df: pd.DataFrame, filename_hint: str = "") -> pd.DataFrame:
        """
        Standardize columns across datasets:
        - Ensure N exists (from run_id if missing)
        - Ensure p_channel exists (fallback to p if needed)
        - Normalize twirling flag column to boolean twirling_enabled if present
        """
        # For steps files, extract N from run_id if not present
        if ('steps' in filename_hint) and ('N' not in df.columns) and ('run_id' in df.columns):
            def extract_N(run_id):
                # Format examples:
                #   M1_N512_dephase_z_iid_p_p0.50000_twirl
                #   M5_N1024_depolarizing_iid_p_p0.30000
                parts = str(run_id).split('_')
                for part in parts:
                    if part.startswith('N'):
                        try:
                            return int(part[1:])
                        except Exception:
                            return None
                return None
            df['N'] = df['run_id'].apply(extract_N)

        # Ensure p_channel exists (GlobalTwirl finals/steps should already have it, but be robust)
        if 'p_channel' not in df.columns:
            if 'p' in df.columns:
                df['p_channel'] = df['p'].astype(float)
            else:
                df['p_channel'] = np.nan

        # Normalize twirling column name and convert to boolean
        if 'twirling_applied' in df.columns:
            df['twirling_enabled'] = df['twirling_applied'].map(
                lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else False
            )
        elif 'twirling_enabled' in df.columns and df['twirling_enabled'].dtype == 'object':
            df['twirling_enabled'] = df['twirling_enabled'].map(
                lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else x
            )

        return df
    
    def plot_fidelity_grid_vs_depth_mini(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        2x2 grid showing fidelity vs purification level for specific p values.
        Each subplot shows curves for different M values at fixed p.
        Now supports three types: 'depolarizing', 'dephasing_untwirled', 'dephasing_twirled'
        """
        # Select data based on noise type
        if noise_type == 'depolarizing':
            df = self.depol_steps
            target_p_values = [0.1, 0.3, 0.7, 0.8]
            title_suffix = "Depolarizing"
            filename_suffix = "depolarizing"
        elif noise_type == 'dephasing_untwirled':
            df = self.dephase_untwirled_steps
            target_p_values = [0.1, 0.3, 0.5, 0.6]
            title_suffix = "Dephasing (Untwirled)"
            filename_suffix = "dephasing_untwirled"
        elif noise_type == 'dephasing_twirled':
            df = self.dephase_twirled_steps
            target_p_values = [0.1, 0.3, 0.7, 0.8]
            title_suffix = "Dephasing (Twirled)"
            filename_suffix = "dephasing_twirled"
        else:
            print(f"Unknown noise type: {noise_type}")
            return None

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Use max N
        max_N = df['N'].max()
        df_N = df[df['N'] == max_N].copy()

        # Find closest p values in data for each target p
        available_ps = df_N['p_channel'].unique()
        ps = []
        for target_p in target_p_values:
            closest_p = min(available_ps, key=lambda x: abs(x - target_p))
            # Only use if within reasonable tolerance (0.05)
            if abs(closest_p - target_p) <= 0.05:
                ps.append(closest_p)
            else:
                print(f"Warning: No data found close to p={target_p:.1f} for {noise_type}")
        
        if len(ps) == 0:
            print(f"No valid p values found for {noise_type}")
            return None

        # Get unique M values
        M_values = sorted(df_N['M'].unique())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        subplot_labels = ['a', 'b', 'c', 'd']

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # fig.suptitle(f'Fidelity Evolution ({title_suffix})', fontsize=28, y=0.95)

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        for plot_idx, p in enumerate(ps):
            if plot_idx >= 4:  # Safety check for 2x2 grid
                break
                
            ax = axes_flat[plot_idx]
            df_p = df_N[df_N['p_channel'] == p].copy()

            # Plot curves for each M value
            for i, M in enumerate(M_values):
                df_M = df_p[df_p['M'] == M].copy()
                
                if len(df_M) > 0:
                    # Group by depth and take max fidelity
                    evolution = df_M.groupby('depth')['fidelity'].max().reset_index()
                    
                    if len(evolution) > 0:
                        ax.plot(evolution['depth'], evolution['fidelity'],
                            linestyle='-', marker=_mk(i),
                            color=colors[i], linewidth=2, markevery=1, markersize=12,
                            label=f'M = {M}', alpha=0.8)

            # Subplot formatting
            ax.set_title(f'$p = {p:.2f}$', fontsize=30)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([2, 4, 6, 8, 10])
            
            # Add legend to top-left subplot
            if plot_idx == 0:
                ax.legend(fontsize=20, loc='lower right', frameon=False)
                
            # Y-axis label only on first column
            if plot_idx == 0 or plot_idx == 2:
                ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
                
            # X-axis label only on bottom row
            if plot_idx == 2 or plot_idx == 3:
                ax.set_xlabel(r'Purification Rounds, $\ell$', fontsize=40)
                
            # Add subplot label 
            if plot_idx == 0:
                ax.text(0.08, 0.04, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='bottom', ha='right')
            else:
                ax.text(0.08, 0.98, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                        fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # Hide unused subplots (if less than 4 p values found)
        for plot_idx in range(len(ps), 4):
            axes_flat[plot_idx].set_visible(False)

        plt.tight_layout()

        filename = f"fidelity_grid_vs_depth_{filename_suffix}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_fidelity_combined_M_mini(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Combined fidelity plots: 3x2 grid with depolarizing (top), untwirled dephasing (middle), 
        and twirled dephasing (bottom). Each column represents M=1 and M=5 only.
        This version plots FIDELITY instead of error rate.
        """
        # Check if we have data for all noise types
        if (self.depol_finals.empty and self.dephase_untwirled_finals.empty and 
            self.dephase_twirled_finals.empty):
            print("No data for fidelity plots")
            return None

        # Create 3x2 subplot grid
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))

        # Noise type configurations
        noise_configs = [
            {
                'type': 'depolarizing',
                'df_finals': self.depol_finals,
                'row_idx': 0,
                'row_label': 'Depolarizing',
                'subplot_label': ['a','b']
            },
            {
                'type': 'dephasing_untwirled', 
                'df_finals': self.dephase_untwirled_finals,
                'row_idx': 1,
                'row_label': 'Dephasing (Untwirled)',
                'subplot_label': ['c','d']
            },
            {
                'type': 'dephasing_twirled', 
                'df_finals': self.dephase_twirled_finals,
                'row_idx': 2,
                'row_label': 'Dephasing (Twirled)',
                'subplot_label': ['e','f']
            }
        ]

        # M values to plot - only M=1 and M=5
        M_values = [1, 5]
        
        for noise_config in noise_configs:
            df = noise_config['df_finals']
            row_idx = noise_config['row_idx']
            noise_type = noise_config['type']
            
            if df.empty:
                # If no data for this noise type, show empty row with message
                for col_idx in range(2):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, f'No {noise_type}\ndata', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, alpha=0.7)
                    if col_idx == 0:
                        ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
                    if row_idx == 2:  # Bottom row
                        ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=40)
                continue

            # Plot each M value
            for col_idx, M in enumerate(M_values):
                ax = axes[row_idx, col_idx]
                
                # Filter for this M value
                df_M = df[df['M'] == M].copy()
                
                # Add subplot label 
                ax.text(0.98, 0.98, noise_config['subplot_label'][col_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='top', ha='right')
                
                if df_M.empty:
                    # If no data for this M, show message
                    ax.text(0.5, 0.5, f'No data\nfor M={M}', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, alpha=0.7)
                else:
                    # Get unique N values for this M
                    N_values = sorted(df_M['N'].unique())
                    colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 
                             'deeppink', 'darkslategrey', 'fuchsia', 'gold']

                    # Plot curves for different N values
                    for i, N in enumerate(N_values):
                        df_N = df_M[df_M['N'] == N].sort_values('p_channel')
                        
                        if len(df_N) > 0:
                            # Convert error to fidelity: fidelity = 1 - eps_L_final
                            fidelity = 1 - df_N['eps_L_final']
                            ax.plot(df_N['p_channel'], fidelity,
                                linestyle='-', marker=_mk(i),
                                color=colors[i], linewidth=2, markersize=12,
                                label=rf'$\ell$ = {int(np.log2(N))}', alpha=0.8)

                    # Set axis limits
                    ax.set_xlim(0.09, 1.0)
                    ax.set_ylim(1e-3, 1.0)
                    ax.set_xscale('linear')
                    ax.set_yscale('linear')
                    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
                    if col_idx == 1:
                        ax.set_yscale('log')
                    
                    # Add legend to one subplot (bottom right)
                    if row_idx == 2 and col_idx == 1 and len(N_values) > 0:
                        ax.legend(fontsize=16, loc='lower left', ncols=2, frameon=False)

                # Subplot titles (M values) only on top row
                if row_idx == 0:
                    ax.set_title(f'M = {M}', fontsize=40)
                
                # Y-axis label only on first column
                if col_idx == 0:
                    ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
                
                # X-axis label only on bottom row
                if row_idx == 2:
                    ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)

        # Add row labels
        fig.text(0.02, 0.83, 'Depolarizing Noise', rotation=90, fontsize=25, 
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.52, 'Untwirled Dephasing', rotation=90, fontsize=25, 
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.21, 'Twirled Dephasing', rotation=90, fontsize=25, 
                verticalalignment='center', weight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.18)  # Make room for row labels

        filename = f"fidelity_combined_M_3rows.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_fidelity_combined_M_mini_inset(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Combined fidelity plots: 3x2 grid with depolarizing (top), untwirled dephasing (middle),
        and twirled dephasing (bottom). Each column represents M=1 and M=5 only.
        This version plots FIDELITY instead of error rate.

        Added: an inset on the top-right subplot (row=0,col=1) = depolarizing, M=5.
            Inset is bottom-left and centered at p=0.8.
        """
        import matplotlib.ticker as mticker
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # --- inset config (only used for depolarizing, M=5 panel) ---
        inset_p_center = 0.80
        inset_p_halfwidth = 0.1          # tweak if you want tighter/wider zoom
        inset_loc = "lower left"
        inset_size = ("45%", "45%")       # (width, height) of inset
        inset_ticksize = 10
        inset_linewidth = 1.8
        inset_markersize = 7
        # Optional fixed y-lims for inset; set to None to auto-pick from data in window.
        inset_ylim = (8e-3, 5e-2)

        # Check if we have data for all noise types
        if (self.depol_finals.empty and self.dephase_untwirled_finals.empty and
            self.dephase_twirled_finals.empty):
            print("No data for fidelity plots")
            return None

        # Create 3x2 subplot grid
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))

        # Noise type configurations
        noise_configs = [
            {
                'type': 'depolarizing',
                'df_finals': self.depol_finals,
                'row_idx': 0,
                'row_label': 'Depolarizing',
                'subplot_label': ['a','b']
            },
            {
                'type': 'dephasing_untwirled',
                'df_finals': self.dephase_untwirled_finals,
                'row_idx': 1,
                'row_label': 'Dephasing (Untwirled)',
                'subplot_label': ['c','d']
            },
            {
                'type': 'dephasing_twirled',
                'df_finals': self.dephase_twirled_finals,
                'row_idx': 2,
                'row_label': 'Dephasing (Twirled)',
                'subplot_label': ['e','f']
            }
        ]

        # M values to plot - only M=1 and M=5
        M_values = [1, 5]

        for noise_config in noise_configs:
            df = noise_config['df_finals']
            row_idx = noise_config['row_idx']
            noise_type = noise_config['type']

            if df.empty:
                # If no data for this noise type, show empty row with message
                for col_idx in range(2):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, f'No {noise_type}\ndata',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=14, alpha=0.7)
                    if col_idx == 0:
                        ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
                    if row_idx == 2:  # Bottom row
                        ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=40)
                continue

            # Plot each M value
            for col_idx, M in enumerate(M_values):
                ax = axes[row_idx, col_idx]

                # Filter for this M value
                df_M = df[df['M'] == M].copy()

                # Add subplot label
                ax.text(0.98, 0.98, noise_config['subplot_label'][col_idx], transform=ax.transAxes, fontsize=28,
                        fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

                if df_M.empty:
                    ax.text(0.5, 0.5, f'No data\nfor M={M}',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12, alpha=0.7)
                else:
                    # Get unique N values for this M
                    N_values = sorted(df_M['N'].unique())
                    colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown',
                            'deeppink', 'darkslategrey', 'fuchsia', 'gold']

                    # Plot curves for different N values
                    for i, N in enumerate(N_values):
                        df_N = df_M[df_M['N'] == N].sort_values('p_channel')
                        if len(df_N) == 0:
                            continue

                        fidelity = 1 - df_N['eps_L_final']
                        ax.plot(
                            df_N['p_channel'], fidelity,
                            linestyle='-', marker=_mk(i),
                            color=colors[i % len(colors)],
                            linewidth=2, markersize=12,
                            label=rf'$\ell$ = {int(np.log2(N))}', alpha=0.8
                        )

                    # Set axis limits/scales
                    ax.set_xlim(0.09, 1.0)
                    ax.set_ylim(1e-3, 1.0)
                    ax.set_xscale('linear')
                    ax.set_yscale('linear')
                    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

                    # Make M=5 column log-y, as before
                    if col_idx == 1:
                        ax.set_yscale('log')

                    # --------- ADD INSET ONLY FOR: depolarizing (row 0) AND M=5 (col 1) ----------
                    if (row_idx == 0) and (col_idx == 1) and (noise_type == "depolarizing"):
                        pmin = max(0.0, inset_p_center - inset_p_halfwidth)
                        pmax = min(1.0, inset_p_center + inset_p_halfwidth)

                        axins = inset_axes(
                            ax,
                            width=inset_size[0],
                            height=inset_size[1],
                            loc=inset_loc,
                            borderpad=1.0
                        )

                        # Plot the same series on inset, but only in [pmin,pmax]
                        for i, N in enumerate(N_values):
                            df_N = df_M[df_M['N'] == N].sort_values('p_channel')
                            if len(df_N) == 0:
                                continue

                            # zoom filter (use isclose-safe comparisons if needed)
                            dzz = df_N[(df_N["p_channel"] >= pmin) & (df_N["p_channel"] <= pmax)].copy()
                            if dzz.empty:
                                continue

                            fidelity_z = 1 - dzz["eps_L_final"].to_numpy(dtype=float)

                            axins.plot(
                                dzz["p_channel"].to_numpy(dtype=float),
                                fidelity_z,
                                linestyle='-',
                                marker=_mk(i),
                                color=colors[i % len(colors)],
                                linewidth=inset_linewidth,
                                markersize=inset_markersize,
                                alpha=0.9
                            )

                        # x/y limits and scales
                        axins.set_xlim(pmin, pmax)

                        # Match parent axis y-scale (log for this panel)
                        axins.set_yscale(ax.get_yscale())

                        # Optional: vertical reference line at p=0.8
                        axins.axvline(inset_p_center, color="black", linewidth=1.2, alpha=0.9)

                        # y-lims: either fixed or auto from data in window
                        if inset_ylim is not None:
                            axins.set_ylim(inset_ylim[0], inset_ylim[1])
                        else:
                            # auto y-lims from the zoomed data (positive only for log)
                            y_all = []
                            for N in N_values:
                                df_N = df_M[df_M['N'] == N].sort_values('p_channel')
                                dzz = df_N[(df_N["p_channel"] >= pmin) & (df_N["p_channel"] <= pmax)]
                                if not dzz.empty:
                                    y_all.append((1 - dzz["eps_L_final"]).to_numpy(dtype=float))
                            if y_all:
                                y_all = np.concatenate(y_all)
                                if axins.get_yscale() == "log":
                                    y_all = y_all[y_all > 0]
                                if y_all.size > 0:
                                    ymin = float(np.min(y_all)) * (0.8 if axins.get_yscale() == "log" else 0.95)
                                    ymax = float(np.max(y_all)) * (1.2 if axins.get_yscale() == "log" else 1.05)
                                    if axins.get_yscale() == "log":
                                        ymin = max(ymin, 1e-6)
                                        ymax = max(ymax, ymin * 1.01)
                                    axins.set_ylim(ymin, ymax)

                        # tick font sizes
                        axins.tick_params(axis="both", which="major", labelsize=inset_ticksize)

                        # if log scale + global rcParams are fighting you, enforce:
                        for t in axins.get_xticklabels(which="both"):
                            t.set_fontsize(inset_ticksize)
                        for t in axins.get_yticklabels(which="both"):
                            t.set_fontsize(inset_ticksize)

                        # keep inset clean
                        axins.grid(False)

                    # Legend only once (bottom right), as before
                    if row_idx == 2 and col_idx == 1 and len(N_values) > 0:
                        ax.legend(fontsize=16, loc='lower left', ncols=2, frameon=False)

                # Subplot titles (M values) only on top row
                if row_idx == 0:
                    ax.set_title(f'M = {M}', fontsize=40)

                # Y-axis label only on first column
                if col_idx == 0:
                    ax.set_ylabel(r'Fidelity, $F$', fontsize=35)

                # X-axis label only on bottom row
                if row_idx == 2:
                    ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)

        # Add row labels
        fig.text(0.02, 0.83, 'Depolarizing Noise', rotation=90, fontsize=25,
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.52, 'Untwirled Dephasing', rotation=90, fontsize=25,
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.21, 'Twirled Dephasing', rotation=90, fontsize=25,
                verticalalignment='center', weight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.18)  # Make room for row labels

        filename = f"fidelity_combined_M_3rows.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)

    
    # ----------------------------
    # NEW: GlobalTwirl fidelity grid
    # ----------------------------
    def plot_fidelity_grid_vs_depth_globalTwirl(self, noise_type: str = "dephase_z", save_format: str = "pdf") -> Optional[str]:
        """
        Create a 2x2 grid plot for GlobalTwirl data:
          - Uses data/globalTwirl_simulations
          - Uses STEPS data (fidelity vs 'depth')
          - Uses max N within that dataset
          - Chooses closest p values to a target list (with tolerance 0.05)
          - One curve per M (same behavior/style as existing mini grid)
        
        noise_type: "dephase_z" or "depolarizing"
        """

        if noise_type == "dephase_z":
            df = self.global_dephase_steps
            target_p_values = [0.1, 0.3, 0.5, 0.6]
            title_suffix = "GlobalTwirl Dephasing (Z)"
            filename_suffix = "globalTwirl_dephase_z"
        else:
            print(f"Unknown GlobalTwirl noise_type: {noise_type}")
            return None

        if df.empty:
            print(f"No GlobalTwirl steps data for {noise_type}")
            return None

        # Use max N available (consistent with your existing function)
        if 'N' not in df.columns or df['N'].isna().all():
            print("GlobalTwirl steps data missing N (and could not extract from run_id).")
            return None
        max_N = int(df['N'].max())
        df_N = df[df['N'] == max_N].copy()

        if df_N.empty:
            print(f"No GlobalTwirl data found at max N={max_N} for {noise_type}")
            return None

        # Find closest p values in data for each target p
        if 'p_channel' not in df_N.columns:
            print("GlobalTwirl steps data missing p_channel.")
            return None
        available_ps = sorted([float(x) for x in df_N['p_channel'].dropna().unique()])
        if len(available_ps) == 0:
            print("No p_channel values available in GlobalTwirl data.")
            return None

        ps = []
        for target_p in target_p_values:
            closest_p = min(available_ps, key=lambda x: abs(x - target_p))
            if abs(closest_p - target_p) <= 0.05:
                ps.append(closest_p)
            else:
                print(f"Warning: No data found close to p={target_p:.1f} for GlobalTwirl {noise_type}")

        if len(ps) == 0:
            print(f"No valid p values found for GlobalTwirl {noise_type}")
            return None

        # M values in this slice
        M_values = sorted([int(m) for m in df_N['M'].dropna().unique()])
        if len(M_values) == 0:
            print("No M values found in GlobalTwirl steps data.")
            return None

        # Preserve your existing color choices for this plot function (do not change global style)
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        subplot_labels = ['a', 'b', 'c', 'd']

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes_flat = axes.flatten()

        for plot_idx, p in enumerate(ps):
            if plot_idx >= 4:
                break

            ax = axes_flat[plot_idx]
            df_p = df_N[np.isclose(df_N['p_channel'].astype(float), float(p))].copy()
            if df_p.empty:
                # fallback exact compare if needed
                df_p = df_N[df_N['p_channel'] == p].copy()

            # Plot curves for each M value
            for i, M in enumerate(M_values):
                df_M = df_p[df_p['M'] == M].copy()
                if len(df_M) == 0:
                    continue

                # Group by depth and take max fidelity (same as your current logic)
                evolution = df_M.groupby('depth')['fidelity'].max().reset_index()
                if len(evolution) == 0:
                    continue

                ax.plot(
                    evolution['depth'], evolution['fidelity'],
                    linestyle='-', marker=_mk(i),
                    color=colors[i % len(colors)], linewidth=2, markevery=1, markersize=12,
                    label=f'M = {M}', alpha=0.8
                )

            # Subplot formatting (keep same knobs as your existing mini grid)
            ax.set_title(f'$p = {float(p):.2f}$', fontsize=30)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([2, 4, 6, 8, 10])

            # Legend only on top-left subplot
            if plot_idx == 0:
                ax.legend(fontsize=20, loc='lower right', frameon=False)

            # Y-axis label only on first column
            if plot_idx == 0 or plot_idx == 2:
                ax.set_ylabel(r'Fidelity, $F$', fontsize=40)

            # X-axis label only on bottom row
            if plot_idx == 2 or plot_idx == 3:
                ax.set_xlabel(r'Purification Rounds, $\ell$', fontsize=40)

            # Subplot label positions (same as your current function)
            if plot_idx == 0:
                ax.text(
                    0.08, 0.04, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28,
                    fontweight='bold', fontfamily='sans-serif', va='bottom', ha='right'
                )
            else:
                ax.text(
                    0.08, 0.98, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28,
                    fontweight='bold', fontfamily='sans-serif', va='top', ha='right'
                )

        # Hide unused subplots if fewer than 4 p values found
        for plot_idx in range(len(ps), 4):
            axes_flat[plot_idx].set_visible(False)

        plt.tight_layout()

        filename = f"fidelity_grid_vs_depth_{filename_suffix}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)

    # def plot_fidelity_combined_M_mini_4x2(self, save_format: str = 'pdf') -> Optional[str]:
    #     """
    #     Combined fidelity plots: 4x2 grid with:
    #     Row 1: depolarizing
    #     Row 2: dephasing (untwirled)
    #     Row 3: dephasing (GLOBAL twirl)
    #     Row 4: dephasing (FULL twirl)

    #     Each column represents M=1 and M=5 only.
    #     This version plots FIDELITY instead of error rate.
    #     """
    #     # Check if we have *any* data
    #     if (self.depol_finals.empty and self.dephase_untwirled_finals.empty and
    #         self.global_dephase_finals.empty and self.dephase_twirled_finals.empty):
    #         print("No data for fidelity plots")
    #         return None

    #     # Create 4x2 subplot grid
    #     fig, axes = plt.subplots(4, 2, figsize=(12, 20))

    #     # Noise type configurations (row order matters)
    #     noise_configs = [
    #         {
    #             'type': 'depolarizing',
    #             'df_finals': self.depol_finals,
    #             'row_idx': 0,
    #             'row_label': 'Depolarizing',
    #             'subplot_label': ['a', 'b']
    #         },
    #         {
    #             'type': 'dephasing_untwirled',
    #             'df_finals': self.dephase_untwirled_finals,
    #             'row_idx': 1,
    #             'row_label': 'Dephasing (Untwirled)',
    #             'subplot_label': ['c', 'd']
    #         },
    #         {
    #             'type': 'dephasing_globalTwirl',
    #             'df_finals': self.global_dephase_finals,
    #             'row_idx': 2,
    #             'row_label': 'Dephasing (Global Twirl)',
    #             'subplot_label': ['e', 'f']
    #         },
    #         {
    #             'type': 'dephasing_fullTwirl',
    #             'df_finals': self.dephase_twirled_finals,
    #             'row_idx': 3,
    #             'row_label': 'Dephasing (Full Twirl)',
    #             'subplot_label': ['g', 'h']
    #         },
    #     ]

    #     # M values to plot - only M=1 and M=5
    #     M_values = [1, 5]

    #     for noise_config in noise_configs:
    #         df = noise_config['df_finals']
    #         row_idx = noise_config['row_idx']
    #         noise_type = noise_config['type']

    #         if df.empty:
    #             # If no data for this noise type, show empty row with message
    #             for col_idx in range(2):
    #                 ax = axes[row_idx, col_idx]
    #                 ax.text(
    #                     0.5, 0.5, f'No {noise_type}\ndata',
    #                     horizontalalignment='center', verticalalignment='center',
    #                     transform=ax.transAxes, fontsize=14, alpha=0.7
    #                 )
    #                 if col_idx == 0:
    #                     ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
    #                 if row_idx == 3:  # Bottom row
    #                     ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)
    #             continue

    #         # Plot each M value
    #         for col_idx, M in enumerate(M_values):
    #             ax = axes[row_idx, col_idx]

    #             # Filter for this M value
    #             df_M = df[df['M'] == M].copy()

    #             # Add subplot label
    #             ax.text(
    #                 0.98, 0.98, noise_config['subplot_label'][col_idx],
    #                 transform=ax.transAxes, fontsize=28,
    #                 fontweight='bold', fontfamily='sans-serif', va='top', ha='right'
    #             )

    #             if df_M.empty:
    #                 ax.text(
    #                     0.5, 0.5, f'No data\nfor M={M}',
    #                     horizontalalignment='center', verticalalignment='center',
    #                     transform=ax.transAxes, fontsize=12, alpha=0.7
    #                 )
    #             else:
    #                 # Get unique N values for this M
    #                 N_values = sorted(df_M['N'].unique())
    #                 colors = [
    #                     'red', 'green', 'blue', 'orange', 'purple', 'saddlebrown',
    #                     'deeppink', 'darkslategrey', 'fuchsia', 'gold'
    #                 ]

    #                 # Plot curves for different N values
    #                 for i, N in enumerate(N_values):
    #                     df_N = df_M[df_M['N'] == N].sort_values('p_channel')
    #                     if len(df_N) == 0:
    #                         continue

    #                     fidelity = 1 - df_N['eps_L_final']
    #                     ax.plot(
    #                         df_N['p_channel'], fidelity,
    #                         linestyle='-', marker=_mk(i),
    #                         color=colors[i % len(colors)], linewidth=2, markersize=12,
    #                         label=rf'$\ell$ = {int(np.log2(N))}', alpha=0.8
    #                     )

    #                 # Axis limits/scales (keep identical to your previous behavior)
    #                 ax.set_xlim(0.09, 1.0)
    #                 ax.set_ylim(1e-3, 1.0)
    #                 ax.set_xscale('linear')
    #                 ax.set_yscale('linear')
    #                 ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

    #                 # Make M=5 column log-y, as before
    #                 if col_idx == 1:
    #                     ax.set_yscale('log')

    #                 # Legend only once (bottom right), as before logic but now bottom row is row_idx==3
    #                 if row_idx == 3 and col_idx == 1 and len(N_values) > 0:
    #                     ax.legend(fontsize=16, loc='lower left', ncols=2, frameon=False)

    #             # Subplot titles (M values) only on top row
    #             if row_idx == 0:
    #                 ax.set_title(f'M = {M}', fontsize=40)

    #             # Y-axis label only on first column
    #             if col_idx == 0:
    #                 ax.set_ylabel(r'Fidelity, $F$', fontsize=35)

    #             # X-axis label only on bottom row
    #             if row_idx == 3:
    #                 ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)

    #     # Add row labels (left margin)
    #     fig.text(0.02, 0.87, 'Depolarizing Noise', rotation=90, fontsize=25,
    #             verticalalignment='center', weight='bold')
    #     fig.text(0.02, 0.64, 'Untwirled Dephasing', rotation=90, fontsize=25,
    #             verticalalignment='center', weight='bold')
    #     fig.text(0.02, 0.41, 'Global-Twirled Dephasing', rotation=90, fontsize=25,
    #             verticalalignment='center', weight='bold')
    #     fig.text(0.02, 0.18, 'Full-Twirled Dephasing', rotation=90, fontsize=25,
    #             verticalalignment='center', weight='bold')

    #     plt.tight_layout()
    #     plt.subplots_adjust(left=0.18)  # Make room for row labels

    #     filename = f"fidelity_combined_M_4rows_globalTwirl.{save_format}"
    #     filepath = self.figures_dir / filename
    #     plt.savefig(filepath, dpi=300, bbox_inches='tight')
    #     plt.close()

    #     print(f"Saved {filename}")
    #     return str(filepath)
    
    
    
    def plot_fidelity_combined_M_mini_4x2(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Combined fidelity plots: 4x2 grid with:
        Row 1: depolarizing
        Row 2: dephasing (untwirled)  
        Row 3: dephasing (GLOBAL twirl) - FIXED to use iteration==1 data
        Row 4: dephasing (FULL twirl)

        Each column represents M=1 and M=5 only.
        This version plots FIDELITY instead of error rate.
        """
        # Check if we have *any* data
        if (self.depol_finals.empty and self.dephase_untwirled_finals.empty and
            self.global_dephase_steps.empty and self.dephase_twirled_finals.empty):
            print("No data for fidelity plots")
            return None

        # Create 4x2 subplot grid
        fig, axes = plt.subplots(4, 2, figsize=(12, 20))

        # Noise type configurations (row order matters)
        noise_configs = [
            {
                'type': 'depolarizing',
                'df_data': self.depol_finals,
                'row_idx': 0,
                'row_label': 'Depolarizing',
                'subplot_label': ['a', 'b'],
                'use_finals': True  # Use final iteration data
            },
            {
                'type': 'dephasing_untwirled',
                'df_data': self.dephase_untwirled_finals,
                'row_idx': 1,
                'row_label': 'Dephasing (Untwirled)',
                'subplot_label': ['c', 'd'],
                'use_finals': True  # Use final iteration data
            },
            # {
            #     'type': 'dephasing_globalTwirl',
            #     'df_data': self.global_dephase_steps,  # FIXED: Use steps data
            #     'row_idx': 2,
            #     'row_label': 'Dephasing (Global Twirl)',
            #     'subplot_label': ['e', 'f'],
            #     'use_finals': False  # FIXED: Use iteration==1 data
            # },
            {
                'type': 'dephasing_subsetTwirl',
                'df_data': self.subsetTwirl_dephase_steps,  # FIXED: Use steps data
                'df_data_debug': self.subsetTwirl_dephase_steps_debug, 
                'row_idx': 2,
                'row_label': 'Dephasing (Subset Twirl)', 
                'subplot_label': ['e', 'f'],
                'use_finals': False  # FIXED: Use iteration==1 data
            },
            {
                'type': 'dephasing_fullTwirl',
                'df_data': self.dephase_twirled_finals,
                'row_idx': 3,
                'row_label': 'Dephasing (Full Twirl)',
                'subplot_label': ['g', 'h'],
                'use_finals': True  # Use final iteration data
            },
        ]

        # M values to plot - only M=1 and M=5
        M_values = [1, 5]

        for noise_config in noise_configs:
            df = noise_config['df_data']
            row_idx = noise_config['row_idx']
            noise_type = noise_config['type']
            use_finals = noise_config['use_finals']
            
            if noise_config['type'] == 'dephasing_subsetTwirl':
                df_debug = noise_config['df_data_debug']

            if df.empty:
                # If no data for this noise type, show empty row with message
                for col_idx in range(2):
                    ax = axes[row_idx, col_idx]
                    ax.text(
                        0.5, 0.5, f'No {noise_type}\ndata',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, alpha=0.7
                    )
                    if col_idx == 0:
                        ax.set_ylabel(r'Fidelity, $F$', fontsize=35)
                    if row_idx == 3:  # Bottom row
                        ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)
                continue

            # FIXED: Different data processing for row 3 (global twirl)
            if not use_finals:
                # For row 3: Filter for iteration==1 from steps data
                df = df[df['iteration'] == 1].copy()
                if noise_config['type'] == 'dephasing_subsetTwirl':
                    df_debug = df_debug[df_debug['iteration'] == 1].copy()

            # Plot each M value
            for col_idx, M in enumerate(M_values):
                ax = axes[row_idx, col_idx]

                # Filter for this M value
                df_M = df[df['M'] == M].copy()
                if noise_config['type'] == 'dephasing_subsetTwirl':
                    df_debug_M = df_debug[df_debug['M'] == M].copy()

                # Add subplot label
                ax.text(
                    0.98, 0.98, noise_config['subplot_label'][col_idx],
                    transform=ax.transAxes, fontsize=28,
                    fontweight='bold', fontfamily='sans-serif', va='top', ha='right'
                )

                if df_M.empty:
                    ax.text(
                        0.5, 0.5, f'No data\nfor M={M}',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, alpha=0.7
                    )
                else:
                    # FIXED: Different column names for steps vs finals data
                    if use_finals:
                        # For finals data (rows 1, 2, 4)
                        N_values = sorted(df_M['N'].unique())
                        p_col = 'p_channel'
                        fidelity_col = 'eps_L_final'  # Will convert: 1 - eps_L_final
                    else:
                        # For steps data (row 3) - adjust column names as needed
                        N_values = sorted(df_M['purification_level'].unique()) if 'purification_level' in df_M.columns else sorted(df_M['N'].unique())
                        p_col = 'p' if 'p' in df_M.columns else 'p_channel'
                        
                        # Use fidelity directly if available, otherwise convert from error
                        if 'fidelity' in df_M.columns:
                            fidelity_col = 'fidelity'
                            convert_from_error = False
                        else:
                            # Assume there's an error column to convert from
                            possible_error_cols = ['eps_L', 'eps_L_final', 'error']
                            fidelity_col = None
                            for col in possible_error_cols:
                                if col in df_M.columns:
                                    fidelity_col = col
                                    break
                            if fidelity_col is None:
                                fidelity_col = 'fidelity'  # Fallback
                            convert_from_error = True

                    colors = [
                        'red', 'green', 'blue', 'orange', 'purple', 'saddlebrown',
                        'deeppink', 'darkslategrey', 'fuchsia', 'gold'
                    ]

                    # Plot curves for different N/purification_level values
                    for i, N in enumerate(N_values):
                        if use_finals:
                            df_N = df_M[df_M['N'] == N].sort_values(p_col)
                            ell_label = int(np.log2(N))
                        else:
                            # For steps data, use purification_level directly
                            if 'purification_level' in df_M.columns:
                                df_N = df_M[df_M['purification_level'] == N].sort_values(p_col)
                                ell_label = N
                                
                                if noise_config['type'] == 'dephasing_subsetTwirl':
                                    df_debug_N = df_debug_M[df_debug_M['purification_level'] == N].sort_values(p_col)
                            else:
                                df_N = df_M[df_M['N'] == N].sort_values(p_col)
                                ell_label = int(np.log2(N)) if N > 0 else N
                                if noise_config['type'] == 'dephasing_subsetTwirl':
                                    df_debug_N = df_debug_M[df_debug_M['N'] == N].sort_values(p_col)

                        if len(df_N) == 0:
                            continue

                        # Calculate fidelity
                        if use_finals or (not use_finals and 'convert_from_error' in locals() and convert_from_error):
                            fidelity = 1 - df_N[fidelity_col]
                        else:
                            fidelity = df_N[fidelity_col]
                            if noise_config['type'] == 'dephasing_subsetTwirl':
                                fidelity_debug = df_debug_N[fidelity_col]

                        if noise_config['type'] == 'dephasing_subsetTwirl' and ell_label == 0:
                            # Skip plotting the N=1 curve for subset twirl dephasing
                            continue
                        elif noise_config['type'] == 'dephasing_subsetTwirl' and ell_label > 0:
                            if M==1:
                                ax.plot(
                                df_debug_N[p_col], fidelity_debug,
                                linestyle='-', marker=_mk(i-1),  # Shift marker index for subset twirl to keep colors consistent
                                color=colors[i-1 % len(colors)], linewidth=2, markersize=12,
                                label=rf'$\ell$ = {ell_label}', alpha=0.8
                                )
                            else:
                                ax.plot(
                                    df_N[p_col], fidelity,
                                    linestyle='-', marker=_mk(i-1),  # Shift marker index for subset twirl to keep colors consistent
                                    color=colors[i-1 % len(colors)], linewidth=2, markersize=12,
                                    label=rf'$\ell$ = {ell_label}', alpha=0.8
                                )
                        else:
                            ax.plot(
                                df_N[p_col], fidelity,
                                linestyle='-', marker=_mk(i),
                                color=colors[i % len(colors)], linewidth=2, markersize=12,
                                label=rf'$\ell$ = {ell_label}', alpha=0.8
                            )

                    # Axis limits/scales (keep identical to your previous behavior)
                    ax.set_xlim(0.09, 1.0)
                    ax.set_ylim(1e-3, 1.0)
                    ax.set_xscale('linear')
                    ax.set_yscale('linear')
                    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

                    # Make M=5 column log-y, as before
                    if col_idx == 1:
                        ax.set_yscale('log')

                    # Legend only once (bottom right)
                    if row_idx == 3 and col_idx == 1 and len(N_values) > 0:
                        ax.legend(fontsize=16, loc='lower left', ncols=2, frameon=False)

                # Subplot titles (M values) only on top row
                if row_idx == 0:
                    ax.set_title(f'M = {M}', fontsize=40)

                # Y-axis label only on first column
                if col_idx == 0:
                    ax.set_ylabel(r'Fidelity, $F$', fontsize=35)

                # X-axis label only on bottom row
                if row_idx == 3:
                    ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=35)

        # Add row labels (left margin)
        fig.text(0.02, 0.87, 'Depolarizing Noise', rotation=90, fontsize=25,
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.64, 'Untwirled Dephasing', rotation=90, fontsize=25,
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.41, 'Approx. Twirled Dephasing', rotation=90, fontsize=25,
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.18, 'Full-Twirled Dephasing', rotation=90, fontsize=25,
                verticalalignment='center', weight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.18)  # Make room for row labels

        filename = f"fidelity_combined_M_4rows_subsetTwirl.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_psuccess_grid_vs_depth_mini(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        2x2 grid showing success probability vs purification level for specific p values.
        Each subplot shows curves for different M values at fixed p.
        Now supports three types: 'depolarizing', 'dephasing_untwirled', 'dephasing_twirled'
        """
        # Select data based on noise type
        if noise_type == 'depolarizing':
            df = self.depol_steps
            target_p_values = [0.1, 0.3, 0.7, 0.8]
            title_suffix = "Depolarizing"
            filename_suffix = "depolarizing"
        elif noise_type == 'dephasing_untwirled':
            df = self.dephase_untwirled_steps
            target_p_values = [0.1, 0.3, 0.5, 0.6]
            title_suffix = "Dephasing (Untwirled)"
            filename_suffix = "dephasing_untwirled"
        elif noise_type == 'dephasing_twirled':
            df = self.dephase_twirled_steps
            target_p_values = [0.1, 0.3, 0.7, 0.8]
            title_suffix = "Dephasing (Twirled)"
            filename_suffix = "dephasing_twirled"
        else:
            print(f"Unknown noise type: {noise_type}")
            return None

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Use max N
        max_N = df['N'].max()
        df_N = df[df['N'] == max_N].copy()

        # Find closest p values in data for each target p
        available_ps = df_N['p_channel'].unique()
        ps = []
        for target_p in target_p_values:
            closest_p = min(available_ps, key=lambda x: abs(x - target_p))
            # Only use if within reasonable tolerance (0.05)
            if abs(closest_p - target_p) <= 0.05:
                ps.append(closest_p)
            else:
                print(f"Warning: No data found close to p={target_p:.1f} for {noise_type}")
        
        if len(ps) == 0:
            print(f"No valid p values found for {noise_type}")
            return None

        # Get unique M values
        M_values = sorted(df_N['M'].unique())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        subplot_labels = ['a', 'b', 'c', 'd']

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        for plot_idx, p in enumerate(ps):
            if plot_idx >= 4:  # Safety check for 2x2 grid
                break
                
            ax = axes_flat[plot_idx]
            df_p = df_N[df_N['p_channel'] == p].copy()

            # Plot curves for each M value
            for i, M in enumerate(M_values):
                df_M = df_p[df_p['M'] == M].copy()
                
                if len(df_M) > 0:
                    # Group by depth and take mean P_success
                    evolution = df_M.groupby('depth')['P_success'].mean().reset_index()
                    
                    if len(evolution) > 0:
                        ax.plot(evolution['depth'], evolution['P_success'],
                            linestyle='-', marker=_mk(i),
                            color=colors[i], linewidth=2, markevery=1, markersize=12,
                            label=f'M = {M}', alpha=0.8)

            # Subplot formatting
            ax.set_title(f'$p = {p:.2f}$', fontsize=30)
            if plot_idx < 2:
                ax.set_ylim(6e-1, 1.0)
            else:
                ax.set_ylim(5e-1, 1.0)
                
            if noise_type == 'dephasing_untwirled' and plot_idx ==1:
                ax.set_ylim(5e-1, 1.0)
            # ax.set_ylim(6e-1, 1.0)
            ax.set_yscale('log')
            ax.set_xticks([2, 4, 6, 8, 10])
            
            # Add legend to top-left subplot
            if plot_idx == 0:
                ax.legend(fontsize=20, loc='lower right', frameon=False)
                
            # Y-axis label only on first column
            if plot_idx == 0 or plot_idx == 2:
                ax.set_ylabel(r'$P_{\mathrm{suc}}$', fontsize=40)
                
            # X-axis label only on bottom row
            if plot_idx == 2 or plot_idx == 3:
                ax.set_xlabel(r'Purification Rounds, $\ell$  ', fontsize=35)
                
            # Add subplot label 
            if plot_idx == 0:
                ax.text(0.08, 0.04, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='bottom', ha='right')
            else:
                ax.text(0.08, 0.98, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                        fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # Hide unused subplots (if less than 4 p values found)
        for plot_idx in range(len(ps), 4):
            axes_flat[plot_idx].set_visible(False)

        plt.tight_layout()

        filename = f"psuccess_grid_vs_depth_{filename_suffix}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_compound_psuccess_grid_vs_depth_mini(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        2x2 grid showing COMPOUND success probability vs purification level for specific p values.
        
        Compound probability = ∏_{i=1}^{ℓ} P_success(i)
        
        This represents the probability that ALL merges from level 1 to ℓ succeed.
        Unlike individual P_success which increases with ℓ, compound P_success decreases
        exponentially as more merges compound their failure probabilities.
        
        Each subplot shows curves for different M values at fixed p.
        Now supports three types: 'depolarizing', 'dephasing_untwirled', 'dephasing_twirled'
        """
        # Select data based on noise type
        if noise_type == 'depolarizing':
            df = self.depol_steps
            target_p_values = [0.1, 0.3, 0.7, 0.8]
            title_suffix = "Depolarizing"
            filename_suffix = "depolarizing"
        elif noise_type == 'dephasing_untwirled':
            df = self.dephase_untwirled_steps
            target_p_values = [0.1, 0.3, 0.5, 0.6]
            title_suffix = "Dephasing (Untwirled)"
            filename_suffix = "dephasing_untwirled"
        elif noise_type == 'dephasing_twirled':
            df = self.dephase_twirled_steps
            target_p_values = [0.1, 0.3, 0.7, 0.8]
            title_suffix = "Dephasing (Twirled)"
            filename_suffix = "dephasing_twirled"
        else:
            print(f"Unknown noise type: {noise_type}")
            return None

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Use max N
        max_N = df['N'].max()
        df_N = df[df['N'] == max_N].copy()

        # Find closest p values in data for each target p
        available_ps = df_N['p_channel'].unique()
        ps = []
        for target_p in target_p_values:
            closest_p = min(available_ps, key=lambda x: abs(x - target_p))
            # Only use if within reasonable tolerance (0.05)
            if abs(closest_p - target_p) <= 0.05:
                ps.append(closest_p)
            else:
                print(f"Warning: No data found close to p={target_p:.1f} for {noise_type}")
        
        if len(ps) == 0:
            print(f"No valid p values found for {noise_type}")
            return None

        # Get unique M values
        M_values = sorted(df_N['M'].unique())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        subplot_labels = ['a', 'b', 'c', 'd']

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        for plot_idx, p in enumerate(ps):
            if plot_idx >= 4:  # Safety check for 2x2 grid
                break
                
            ax = axes_flat[plot_idx]
            df_p = df_N[df_N['p_channel'] == p].copy()

            # Plot curves for each M value
            for i, M in enumerate(M_values):
                df_M = df_p[df_p['M'] == M].copy()
                
                if len(df_M) > 0:
                    # Group by depth and take mean P_success at each level
                    evolution = df_M.groupby('depth')['P_success'].mean().reset_index()
                    
                    if len(evolution) > 0:
                        # Compute COMPOUND probability (cumulative product)
                        evolution['P_compound'] = evolution['P_success'].cumprod()
                        
                        ax.plot(evolution['depth'], evolution['P_compound'],
                            linestyle='-', marker=_mk(i),
                            color=colors[i], linewidth=2, markevery=1, markersize=12,
                            label=f'M = {M}', alpha=0.8)

            # Subplot formatting
            ax.set_title(f'$p = {p:.2f}$', fontsize=30)
            ax.set_ylim(1e-4, 1.0)
            ax.set_yscale('log')
            ax.set_xticks([2, 4, 6, 8, 10])
            
            # Add legend to top-left subplot
            if plot_idx == 0:
                ax.legend(fontsize=20, loc='lower right', frameon=False)
                
            # Y-axis label only on first column
            if plot_idx == 0 or plot_idx == 2:
                ax.set_ylabel(r'$\prod_{\ell}$  $P_{\mathrm{suc}}$', fontsize=40)
                
            # X-axis label only on bottom row
            if plot_idx == 2 or plot_idx == 3:
                ax.set_xlabel(r'Purification Rounds, $\ell$', fontsize=40)
                
            # Add subplot label 
            if plot_idx == 0:
                ax.text(0.08, 0.04, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='bottom', ha='right')
            else:
                ax.text(0.08, 0.12, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                        fontweight='bold', fontfamily='sans-serif', va='top', ha='right')

        # Hide unused subplots (if less than 4 p values found)
        for plot_idx in range(len(ps), 4):
            axes_flat[plot_idx].set_visible(False)

        plt.tight_layout()

        filename = f"compound_psuccess_grid_vs_depth_{filename_suffix}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_resource_cost_heatmap(self, noise_type: str, M_value: int = 5, save_format: str = 'pdf') -> Optional[str]:
        """
        Create heatmap of log10(C_ℓ) vs (p, ℓ) showing expected resource cost.
        
        C_ℓ = expected number of raw copies needed to produce one ℓ-round purified output
        
        Formula: C_ℓ = 2^ℓ / ∏_{k=1}^{ℓ} P_success(k)
        
        where P_success(k) is the success probability at purification level k.
        
        Parameters
        ----------
        noise_type : str
            'depolarizing' or 'dephasing_untwirled'
        M_value : int
            System size to analyze (default: 5)
        save_format : str
            Output format (pdf, png, etc.)
        """
        # Select data based on noise type
        if noise_type == 'depolarizing':
            df = self.depol_steps
            title_suffix = "Depolarizing Noise"
            filename_suffix = "depolarizing"
        elif noise_type == 'dephasing_untwirled':
            df = self.dephase_untwirled_steps
            title_suffix = "Dephasing (Untwirled)"
            filename_suffix = "dephasing_untwirled"
        else:
            print(f"Unknown noise type: {noise_type}")
            return None

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Filter for specific M and use max N
        df_M = df[df['M'] == M_value].copy()
        
        if df_M.empty:
            print(f"No data for M={M_value}")
            return None
        
        max_N = df_M['N'].max()
        df_M = df_M[df_M['N'] == max_N].copy()

        # Get unique p and depth values
        p_values = np.sort(df_M['p_channel'].unique())
        depth_values = np.sort(df_M['depth'].unique())
        
        print(f"Computing resource costs for {len(p_values)} p values and {len(depth_values)} depths...")

        # Create 2D grid for log10(C_ℓ)
        C_grid = np.full((len(depth_values), len(p_values)), np.nan)
        
        for i, ℓ in enumerate(depth_values):
            for j, p in enumerate(p_values):
                # Get P_success values from depth 1 to ℓ
                df_subset = df_M[(df_M['p_channel'] == p) & 
                                 (df_M['depth'] >= 1) & 
                                 (df_M['depth'] <= ℓ)].copy()
                
                if len(df_subset) > 0:
                    # Group by depth and compute mean P_success at each level
                    p_success_by_depth = df_subset.groupby('depth')['P_success'].mean()
                    
                    # Only proceed if we have data for all depths from 1 to ℓ
                    required_depths = set(range(1, int(ℓ) + 1))
                    available_depths = set(p_success_by_depth.index)
                    
                    if required_depths.issubset(available_depths):
                        # Extract P_success values in order from depth 1 to ℓ
                        p_success_values = [p_success_by_depth[d] for d in range(1, int(ℓ) + 1)]
                        
                        # Compute product ∏_{k=1}^{ℓ} s_k
                        product = np.prod(p_success_values)
                        
                        if product > 0:
                            # C_ℓ = 2^ℓ / ∏ s_k
                            C_ℓ = (2**ℓ) / product
                            C_grid[i, j] = np.log10(C_ℓ)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create meshgrid for pcolormesh
        P, L = np.meshgrid(p_values, depth_values)
        
        # Plot heatmap
        im = ax.pcolormesh(P, L, C_grid, shading='auto', cmap='viridis', vmin=0)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r'$\log_{10}(C_\ell)$', fontsize=40, rotation=270, labelpad=50)
        cbar.ax.tick_params(labelsize=28)
        
        # Add contour lines for key values
        # contour_levels = [1, 2, 3, 4]  # log10(C) = 1,2,3,4 → C = 10,100,1000,10000
        # CS = ax.contour(P, L, C_grid, levels=contour_levels, colors='white', 
        #                linewidths=1.5, alpha=0.6)
        # ax.clabel(CS, inline=True, fontsize=16, fmt='%g')
        
        # Labels and title
        ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=40)
        ax.set_ylabel(r'Purification Rounds, $\ell$', fontsize=40)
        # ax.set_title(f'{title_suffix} ($M={M_value}$)', fontsize=40)
        
        # Set axis limits
        ax.set_xlim(p_values.min(), p_values.max())
        ax.set_ylim(depth_values.min(), depth_values.max())
        
        # Tick parameters
        ax.tick_params(axis='both', which='major', labelsize=28)
        
        plt.tight_layout()

        filename = f"resource_cost_heatmap_{filename_suffix}_M{M_value}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_resource_cost_heatmap_combined(self, M_depol: int = 5, M_dephase: int = 1, 
                                            save_format: str = 'pdf') -> Optional[str]:
        """
        Create side-by-side heatmaps of log10(C_ℓ) for both noise models.
        
        Left panel: Depolarizing noise
        Right panel: Dephasing (untwirled) noise
        """
        # Check if we have both datasets
        if self.depol_steps.empty or self.dephase_untwirled_steps.empty:
            print("Missing data for combined plot")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        noise_configs = [
            {
                'df': self.depol_steps,
                'M': M_depol,
                'title': f'Depolarizing Noise ($M={M_depol}$)',
                'ax': axes[0],
                'label': 'a',
                'panel_id': 0  # Add unique ID
            },
            {
                'df': self.dephase_untwirled_steps,
                'M': M_dephase,
                'title': f'Dephasing Untwirled ($M={M_dephase}$)',
                'ax': axes[1],
                'label': 'b',
                'panel_id': 1  # Add unique ID
            }
        ]
        
        vmin_global = np.inf
        vmax_global = -np.inf
        plot_data = []
        
        # First pass: compute all grids and find global min/max for consistent colorbar
        for config in noise_configs:
            df = config['df']
            M_value = config['M']
            
            # Filter data
            df_M = df[df['M'] == M_value].copy()
            if df_M.empty:
                continue
                
            max_N = df_M['N'].max()
            df_M = df_M[df_M['N'] == max_N].copy()
            
            # Get unique p and depth values
            p_values = np.sort(df_M['p_channel'].unique())
            depth_values = np.sort(df_M['depth'].unique())
            
            # Create grid
            C_grid = np.full((len(depth_values), len(p_values)), np.nan)
            
            for i, ℓ in enumerate(depth_values):
                for j, p in enumerate(p_values):
                    df_subset = df_M[(df_M['p_channel'] == p) & 
                                     (df_M['depth'] >= 1) & 
                                     (df_M['depth'] <= ℓ)].copy()
                    
                    if len(df_subset) > 0:
                        p_success_by_depth = df_subset.groupby('depth')['P_success'].mean()
                        required_depths = set(range(1, int(ℓ) + 1))
                        available_depths = set(p_success_by_depth.index)
                        
                        if required_depths.issubset(available_depths):
                            p_success_values = [p_success_by_depth[d] for d in range(1, int(ℓ) + 1)]
                            product = np.prod(p_success_values)
                            
                            if product > 0:
                                C_ℓ = (2**ℓ) / product
                                C_grid[i, j] = np.log10(C_ℓ)
            
            # Track global min/max
            valid_values = C_grid[~np.isnan(C_grid)]
            if len(valid_values) > 0:
                vmin_global = min(vmin_global, valid_values.min())
                vmax_global = max(vmax_global, valid_values.max())
            
            # Store for second pass
            plot_data.append({
                'config': config,
                'p_values': p_values,
                'depth_values': depth_values,
                'C_grid': C_grid
            })
        
        # Second pass: plot with consistent colorscale
        for idx, data in enumerate(plot_data):  # Use enumerate to get index
            config = data['config']
            ax = config['ax']
            p_values = data['p_values']
            depth_values = data['depth_values']
            C_grid = data['C_grid']
            
            # Create meshgrid
            P, L = np.meshgrid(p_values, depth_values)
            
            # Plot heatmap with global colorscale
            im = ax.pcolormesh(P, L, C_grid, shading='auto', cmap='viridis',
                              vmin=max(0, vmin_global), vmax=vmax_global)
            
            # Add contour lines
            # contour_levels = [1, 2, 3, 4]
            # CS = ax.contour(P, L, C_grid, levels=contour_levels, colors='white',
            #                linewidths=1.5, alpha=0.6)
            # ax.clabel(CS, inline=True, fontsize=16, fmt='%g')
            
            # Labels
            ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=40)
            if idx == 0:  # Left panel only - FIXED: use idx instead of config comparison
                ax.set_ylabel(r'Purification Rounds, $\ell$', fontsize=40)
            ax.set_title(config['title'], fontsize=40)
            
            # Subplot label
            ax.text(0.05, 0.95, config['label'], transform=ax.transAxes, fontsize=28,
                   fontweight='bold', fontfamily='sans-serif', va='top', ha='left',
                   color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.set_xlim(p_values.min(), p_values.max())
            ax.set_ylim(depth_values.min(), depth_values.max())
        
        # Add single colorbar for both panels
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'$\log_{10}(C_\ell)$', fontsize=40, rotation=270, labelpad=50)
        cbar.ax.tick_params(labelsize=28)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        
        filename = f"resource_cost_heatmap_combined.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_resource_cost_heatmap_2x2_grid(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Create 2x2 grid of log10(C_ℓ) heatmaps comparing noise types and system sizes.
        
        Layout:
        - Top row: Depolarizing noise (M=1, M=5)
        - Bottom row: Dephasing untwirled noise (M=1, M=5)
        - Left column: M=1
        - Right column: M=5
        """
        # Check if we have both datasets
        if self.depol_steps.empty or self.dephase_untwirled_steps.empty:
            print("Missing data for 2x2 grid plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # Configuration for all 4 panels
        noise_configs = [
            {
                'df': self.depol_steps,
                'M': 1,
                'title': 'Depolarizing,\n $M=1$',
                'row': 0,
                'col': 0,
                'label': 'a'
            },
            {
                'df': self.depol_steps,
                'M': 5,
                'title': 'Depolarizing,\n $M=5$',
                'row': 0,
                'col': 1,
                'label': 'b'
            },
            {
                'df': self.dephase_untwirled_steps,
                'M': 1,
                'title': 'Untwirled Dephasing,\n $M=1$',
                'row': 1,
                'col': 0,
                'label': 'c'
            },
            {
                'df': self.dephase_untwirled_steps,
                'M': 5,
                'title': 'Untwirled Dephasing,\n $M=5$',
                'row': 1,
                'col': 1,
                'label': 'd'
            }
        ]
        
        vmin_global = np.inf
        vmax_global = -np.inf
        plot_data = []
        
        # First pass: compute all grids and find global min/max for consistent colorbar
        print("Computing resource costs for 2x2 grid...")
        for config in noise_configs:
            df = config['df']
            M_value = config['M']
            
            # Filter data
            df_M = df[df['M'] == M_value].copy()
            if df_M.empty:
                print(f"Warning: No data for M={M_value}")
                continue
                
            max_N = df_M['N'].max()
            df_M = df_M[df_M['N'] == max_N].copy()
            
            # Get unique p and depth values
            p_values = np.sort(df_M['p_channel'].unique())
            depth_values = np.sort(df_M['depth'].unique())
            
            print(f"  {config['title']}: {len(p_values)} p values, {len(depth_values)} depths")
            
            # Create grid
            C_grid = np.full((len(depth_values), len(p_values)), np.nan)
            
            for i, ℓ in enumerate(depth_values):
                for j, p in enumerate(p_values):
                    df_subset = df_M[(df_M['p_channel'] == p) & 
                                     (df_M['depth'] >= 1) & 
                                     (df_M['depth'] <= ℓ)].copy()
                    
                    if len(df_subset) > 0:
                        p_success_by_depth = df_subset.groupby('depth')['P_success'].mean()
                        required_depths = set(range(1, int(ℓ) + 1))
                        available_depths = set(p_success_by_depth.index)
                        
                        if required_depths.issubset(available_depths):
                            p_success_values = [p_success_by_depth[d] for d in range(1, int(ℓ) + 1)]
                            product = np.prod(p_success_values)
                            
                            if product > 0:
                                C_ℓ = (2**ℓ) / product
                                C_grid[i, j] = np.log10(C_ℓ)
            
            # Track global min/max
            valid_values = C_grid[~np.isnan(C_grid)]
            if len(valid_values) > 0:
                vmin_global = min(vmin_global, valid_values.min())
                vmax_global = max(vmax_global, valid_values.max())
            
            # Store for second pass
            plot_data.append({
                'config': config,
                'p_values': p_values,
                'depth_values': depth_values,
                'C_grid': C_grid
            })
        
        # Second pass: plot with consistent colorscale
        for data in plot_data:
            config = data['config']
            row = config['row']
            col = config['col']
            ax = axes[row, col]
            
            p_values = data['p_values']
            depth_values = data['depth_values']
            C_grid = data['C_grid']
            
            # Create meshgrid
            P, L = np.meshgrid(p_values, depth_values)
            
            # Plot heatmap with global colorscale
            im = ax.pcolormesh(P, L, C_grid, shading='auto', cmap='viridis',
                              vmin=max(0, vmin_global), vmax=vmax_global)
            
            # Add contour lines
            # contour_levels = [1, 2, 3, 4]
            # CS = ax.contour(P, L, C_grid, levels=contour_levels, colors='white',
            #                linewidths=1.5, alpha=0.6)
            # ax.clabel(CS, inline=True, fontsize=16, fmt='%g')
            
            # Labels
            # X-axis label only on bottom row
            if row == 1:
                ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=55)
            
            # Y-axis label only on left column
            if col == 0:
                ax.set_ylabel(r'Purification Rounds, $\ell$', fontsize=55)
            
            # Title
            ax.set_title(config['title'], fontsize=50)
            
            # Subplot label
            ax.text(0.05, 0.95, config['label'], transform=ax.transAxes, fontsize=30,
                   fontweight='bold', fontfamily='sans-serif', va='top', ha='left',
                   color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.set_xlim(p_values.min(), p_values.max())
            ax.set_ylim(depth_values.min(), depth_values.max())
        
        # Add single colorbar for all panels
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'$\log_{10}(C_\ell)$', fontsize=55, rotation=270, labelpad=50)
        cbar.ax.tick_params(labelsize=28)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.92)
        
        filename = f"resource_cost_heatmap_2x2_grid.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_swap_count_heatmap_2x2_grid(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Create 2x2 grid of log10(G_ℓ) heatmaps showing expected SWAP test count.
        
        G_ℓ = expected number of SWAP tests required to produce one ℓ-round purified output
        
        Recursion: G_0 = 0, G_{k+1} = (2*G_k + 1) / P_success(k+1)
        
        Layout:
        - Top row: Depolarizing noise (M=1, M=5)
        - Bottom row: Dephasing untwirled noise (M=1, M=5)
        - Left column: M=1
        - Right column: M=5
        """
        # Check if we have both datasets
        if self.depol_steps.empty or self.dephase_untwirled_steps.empty:
            print("Missing data for 2x2 grid plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # Configuration for all 4 panels
        noise_configs = [
            {
                'df': self.depol_steps,
                'M': 1,
                'title': 'Depolarizing,\n $M=1$',
                'row': 0,
                'col': 0,
                'label': 'a'
            },
            {
                'df': self.depol_steps,
                'M': 5,
                'title': 'Depolarizing,\n $M=5$',
                'row': 0,
                'col': 1,
                'label': 'b'
            },
            {
                'df': self.dephase_untwirled_steps,
                'M': 1,
                'title': 'Untwirled Dephasing,\n $M=1$',
                'row': 1,
                'col': 0,
                'label': 'c'
            },
            {
                'df': self.dephase_untwirled_steps,
                'M': 5,
                'title': 'Untwirled Dephasing,\n $M=5$',
                'row': 1,
                'col': 1,
                'label': 'd'
            }
        ]
        
        vmin_global = np.inf
        vmax_global = -np.inf
        plot_data = []
        
        # First pass: compute all grids and find global min/max for consistent colorbar
        print("Computing SWAP test counts for 2x2 grid...")
        for config in noise_configs:
            df = config['df']
            M_value = config['M']
            
            # Filter data
            df_M = df[df['M'] == M_value].copy()
            if df_M.empty:
                print(f"Warning: No data for M={M_value}")
                continue
                
            max_N = df_M['N'].max()
            df_M = df_M[df_M['N'] == max_N].copy()
            
            # Get unique p and depth values
            p_values = np.sort(df_M['p_channel'].unique())
            depth_values = np.sort(df_M['depth'].unique())
            
            print(f"  {config['title']}: {len(p_values)} p values, {len(depth_values)} depths")
            
            # Create grid
            G_grid = np.full((len(depth_values), len(p_values)), np.nan)
            
            for i, ℓ in enumerate(depth_values):
                for j, p in enumerate(p_values):
                    # Get P_success values from depth 1 to ℓ
                    df_subset = df_M[(df_M['p_channel'] == p) & 
                                     (df_M['depth'] >= 1) & 
                                     (df_M['depth'] <= ℓ)].copy()
                    
                    if len(df_subset) > 0:
                        p_success_by_depth = df_subset.groupby('depth')['P_success'].mean()
                        required_depths = set(range(1, int(ℓ) + 1))
                        available_depths = set(p_success_by_depth.index)
                        
                        if required_depths.issubset(available_depths):
                            # Compute G_k recursively
                            # G_0 = 0
                            # G_{k+1} = (2*G_k + 1) / P_success(k+1)
                            G_k = 0.0  # G_0 = 0
                            
                            for k in range(int(ℓ)):
                                depth_idx = k + 1  # depth 1, 2, 3, ...
                                P_k = p_success_by_depth[depth_idx]
                                
                                if P_k > 0:
                                    G_k = (2 * G_k + 1) / P_k
                                else:
                                    G_k = np.nan
                                    break
                            
                            if not np.isnan(G_k) and G_k > 0:
                                G_grid[i, j] = np.log10(G_k)
            
            # Track global min/max
            valid_values = G_grid[~np.isnan(G_grid)]
            if len(valid_values) > 0:
                vmin_global = min(vmin_global, valid_values.min())
                vmax_global = max(vmax_global, valid_values.max())
            
            # Store for second pass
            plot_data.append({
                'config': config,
                'p_values': p_values,
                'depth_values': depth_values,
                'G_grid': G_grid
            })
        
        # Second pass: plot with consistent colorscale
        for data in plot_data:
            config = data['config']
            row = config['row']
            col = config['col']
            ax = axes[row, col]
            
            p_values = data['p_values']
            depth_values = data['depth_values']
            G_grid = data['G_grid']
            
            # Create meshgrid
            P, L = np.meshgrid(p_values, depth_values)
            
            # Plot heatmap with global colorscale
            im = ax.pcolormesh(P, L, G_grid, shading='auto', cmap='viridis',
                              vmin=max(0, vmin_global), vmax=vmax_global)
            
            # Add contour lines
            # contour_levels = [1, 2, 3, 4]
            # CS = ax.contour(P, L, G_grid, levels=contour_levels, colors='white',
            #                linewidths=1.5, alpha=0.6)
            # ax.clabel(CS, inline=True, fontsize=16, fmt='%g')
            
            # Labels
            # X-axis label only on bottom row
            if row == 1:
                ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=55)
            
            # Y-axis label only on left column
            if col == 0:
                ax.set_ylabel(r'Purification Rounds, $\ell$', fontsize=55)
            
            # Title
            ax.set_title(config['title'], fontsize=50)
            
            # Subplot label
            ax.text(0.05, 0.95, config['label'], transform=ax.transAxes, fontsize=30,
                   fontweight='bold', fontfamily='sans-serif', va='top', ha='left',
                   color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.set_xlim(p_values.min(), p_values.max())
            ax.set_ylim(depth_values.min(), depth_values.max())
        
        # Add single colorbar for all panels
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'$\log_{10}(G_\ell)$', fontsize=55, rotation=270, labelpad=50)
        cbar.ax.tick_params(labelsize=28)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.92)
        
        filename = f"swap_count_heatmap_2x2_grid.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









































    
    def generate_all_plots(self, save_format: str = 'pdf') -> Dict[str, Optional[str]]:
        """Generate all figures."""
        print("\n" + "="*70)
        print("GENERATING SIMULATION FIGURES")
        print("="*70)
        
        plots = {}
        
        # Fidelity grid vs depth for all three noise types
        print("\n1. Fidelity grid vs purification level...")
        plots['fidelity_grid_depol'] = self.plot_fidelity_grid_vs_depth_mini('depolarizing', save_format)
        plots['fidelity_grid_dephase_untwirled'] = self.plot_fidelity_grid_vs_depth_mini('dephasing_untwirled', save_format)
        plots['fidelity_grid_dephase_twirled'] = self.plot_fidelity_grid_vs_depth_mini('dephasing_twirled', save_format)
        
        print("\n2. Combined fidelity plot (4x2 grid: all noise types vs M=1,5)...")
        plots['fidelity_combined_M_3rows'] = self.plot_fidelity_combined_M_mini(save_format)
        # plots['fidelity_combined_M_3rows_inset'] = self.plot_fidelity_combined_M_mini_inset(save_format)
        plots['fidelity_combined_M_4rows_globalTwirl'] = self.plot_fidelity_combined_M_mini_4x2(save_format) # NEW
        
        
        # NEW: Fidelity grid vs depth for GlobalTwirl datasets
        print("\n1b. Fidelity grid vs purification level (GlobalTwirl datasets)...")
        plots['fidelity_grid_globalTwirl_dephase_z'] = self.plot_fidelity_grid_vs_depth_globalTwirl("dephase_z", save_format)
        plots['fidelity_grid_globalTwirl_depolarizing'] = self.plot_fidelity_grid_vs_depth_globalTwirl("depolarizing", save_format)
        
        print("\n3. Success probability grid vs purification level...")
        plots['psuccess_grid_depol'] = self.plot_psuccess_grid_vs_depth_mini('depolarizing', save_format)
        plots['psuccess_grid_dephase_untwirled'] = self.plot_psuccess_grid_vs_depth_mini('dephasing_untwirled', save_format)
        plots['psuccess_grid_dephase_twirled'] = self.plot_psuccess_grid_vs_depth_mini('dephasing_twirled', save_format)
        
        print("\n4. Compound success probability grid vs purification level...")
        plots['compound_psuccess_grid_depol'] = self.plot_compound_psuccess_grid_vs_depth_mini('depolarizing', save_format)
        plots['compound_psuccess_grid_dephase_untwirled'] = self.plot_compound_psuccess_grid_vs_depth_mini('dephasing_untwirled', save_format)
        plots['compound_psuccess_grid_dephase_twirled'] = self.plot_compound_psuccess_grid_vs_depth_mini('dephasing_twirled', save_format)
        
        print("\n5. Resource cost heatmaps...")
        # plots['resource_cost_heatmap_depol'] = self.plot_resource_cost_heatmap('depolarizing', M_value=5, save_format=save_format)
        # plots['resource_cost_heatmap_dephase_untwirled'] = self.plot_resource_cost_heatmap('dephasing_untwirled', M_value=5, save_format=save_format)
        # plots['resource_cost_heatmap_combined'] = self.plot_resource_cost_heatmap_combined(M_depol=5, M_dephase=5, save_format=save_format)
        plots['resource_cost_heatmap_2x2_grid'] = self.plot_resource_cost_heatmap_2x2_grid(save_format=save_format)
        plots['swap_count_heatmap_2x2_grid'] = self.plot_swap_count_heatmap_2x2_grid(save_format=save_format)
        
        # Summary
        successful = [name for name, path in plots.items() if path is not None]
        print(f"\n{len(successful)}/{len(plots)} plots generated successfully")
        print(f"Figures saved to: {self.figures_dir}")
        
        return plots


def main():
    """Main function."""
    import sys
    
    data_dir = "data/simulations_v2"
    figures_dir = "figures/results_v4_sim"
    save_format = "pdf"
    
    if '--data-dir' in sys.argv:
        idx = sys.argv.index('--data-dir')
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    
    if '--figures-dir' in sys.argv:
        idx = sys.argv.index('--figures-dir')
        if idx + 1 < len(sys.argv):
            figures_dir = sys.argv[idx + 1]
    
    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        if idx + 1 < len(sys.argv):
            save_format = sys.argv[idx + 1]
    
    plotter = SimulationPlotter(data_dir, figures_dir)
    plots = plotter.generate_all_plots(save_format)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return plots


if __name__ == "__main__":
    main()