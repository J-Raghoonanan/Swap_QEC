"""
Figure generation for Virtual Distillation simulation data.
Loads CSV data from VD simulations and creates publication-quality figures.

Implements 3 specific plotting functions with exact formatting from reference scripts:
1. plot_fidelity_vs_purification_rounds_iteration1 (2x2 grid, iteration==1)
2. plot_fidelity_combined_M_mini_4x2 (4x2 grid, fidelity vs p)
3. plot_combined_fidelity_and_gamma_2x2_grid (2x2 grid, fidelity+gamma)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# Chosen for clear distinctness in print
MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'h', '*']

def _mk(i: int) -> str:
    """Return a distinct marker for series index i (wraps automatically)."""
    return MARKERS[i % len(MARKERS)]


class VirtualDistillationPlotter:
    """Generate figures from Virtual Distillation CSV data."""
    
    def __init__(self, data_dir: str = "data/VD_sim", 
                 figures_dir: str = "figures/VD_results"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load 4 datasets as specified
        self.depol_steps = self._load_csv('steps_vd_depolarizing.csv')
        self.depol_finals = self._load_csv('finals_vd_depolarizing.csv')
        
        self.dephase_untwirled_steps = self._load_csv('steps_vd_dephase_z_no_twirl.csv')
        self.dephase_untwirled_finals = self._load_csv('finals_vd_dephase_z_no_twirl.csv')
        
        self.dephase_twirled_steps = self._load_csv('steps_vd_dephase_z_twirled.csv')
        self.dephase_twirled_finals = self._load_csv('finals_vd_dephase_z_twirled.csv')
        
        self.dephase_theta_phi_steps = self._load_csv('steps_vd_dephase_z_theta_phi_no_twirl.csv')
        self.dephase_theta_phi_finals = self._load_csv('finals_vd_dephase_z_theta_phi_no_twirl.csv')
        
        self.dephase_approx_steps = self._load_csv('steps_vd_dephase_z_subset0.20.csv')
        self.dephase_approx_finals = self._load_csv('finals_vd_dephase_z_subset0.20.csv')
        # self.dephase_approx_steps = self._load_csv('steps_vd_dephase_z_twirl.csv')
        # self.dephase_approx_finals = self._load_csv('finals_vd_dephase_z_twirl.csv')
        
        print(f"Loaded VD data:")
        print(f"  Depolarizing steps: {len(self.depol_steps)} rows")
        print(f"  Dephasing untwirled steps: {len(self.dephase_untwirled_steps)} rows")
        print(f"  Dephasing theta-phi steps: {len(self.dephase_theta_phi_steps)} rows")
        print(f"  Dephasing approx twirled steps: {len(self.dephase_approx_steps)} rows")
    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file if it exists."""
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            
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
        else:
            print(f"Warning: {filename} not found")
            return pd.DataFrame()
    
    def plot_fidelity_vs_purification_rounds_iteration1(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        2x2 grid showing fidelity vs purification rounds (ℓ) for iteration==1.
        Each subplot shows curves for different M values at fixed p.
        Uses the same formatting as plot_fidelity_grid_vs_depth_mini.
        """
        # Select data based on noise type
        if noise_type == 'depolarizing':
            df = self.depol_steps
            target_p_values = [0.1, 0.3, 0.7, 0.8]
            title_suffix = "Depolarizing"
            filename_suffix = "depolarizing"
            twirling_filter = False
        elif noise_type == 'untwirled_dephasing':
            df = self.dephase_untwirled_steps
            # df = self.dephase_theta_phi_steps
            target_p_values = [0.1, 0.3, 0.5, 0.7]
            title_suffix = "Untwirled Dephasing"
            filename_suffix = "dephasing_no_twirl"
            twirling_filter = False
        elif noise_type == 'twirled_dephasing':
            df = self.dephase_twirled_steps
            target_p_values = [0.1, 0.3, 0.5, 0.7]
            title_suffix = "Twirled Dephasing"
            filename_suffix = "dephasing_twirl"
            twirling_filter = True
        elif noise_type == 'theta_phi_dephasing':
            df = self.dephase_theta_phi_steps
            target_p_values = [0.1, 0.3, 0.5, 0.7]
            title_suffix = "Theta-Phi Dephasing"
            filename_suffix = "dephasing_theta_phi"
            twirling_filter = False
        elif noise_type == 'approx_twirled_dephasing':
            df = self.dephase_approx_steps
            target_p_values = [0.1, 0.3, 0.5, 0.7]
            title_suffix = "Approx. Twirled Dephasing"
            filename_suffix = "dephasing_approx_twirl"
            twirling_filter = True
        else:
            print(f"Unknown noise type: {noise_type}")
            return None

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Check for required columns
        required_cols = ['iteration', 'purification_level', 'M', 'p', 'fidelity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None

        # Handle twirling column name and apply filter
        twirling_col = 'twirling_applied' if 'twirling_applied' in df.columns else 'twirling_enabled'
        if twirling_col in df.columns:
            df_filtered = df[df[twirling_col] == twirling_filter].copy()
        else:
            df_filtered = df.copy()

        # Filter for iteration==1
        df_iter1 = df_filtered[df_filtered['iteration'] == 1].copy()
        
        if df_iter1.empty:
            print(f"No iteration==1 data for {noise_type}")
            return None

        # Find closest p values in data for each target p
        available_ps = df_iter1['p'].unique()
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
        M_values = sorted(df_iter1['M'].unique())
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
            df_p = df_iter1[abs(df_iter1['p'] - p) <= 0.01].copy()  # Use small tolerance for p matching

            if df_p.empty:
                ax.text(0.5, 0.5, f'No data\\nfor p = {p:.2f}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14, alpha=0.7)
                continue

            # Plot curves for each M value
            for i, M in enumerate(M_values):
                df_M = df_p[df_p['M'] == M].copy()
                
                if len(df_M) > 0:
                    # Group by purification_level and take mean fidelity (in case of duplicates)
                    evolution = df_M.groupby('purification_level')['fidelity'].mean().reset_index()
                    evolution = evolution.sort_values('purification_level')
                    
                    if len(evolution) > 0:
                        ax.plot(evolution['purification_level'], evolution['fidelity'],
                            linestyle='-', marker=_mk(i),
                            color=colors[i], linewidth=2, markersize=12,
                            label=f'M = {M}', alpha=0.8)

            # Subplot formatting
            ax.set_title(f'$p = {p:.2f}$', fontsize=30)
            ax.set_ylim(0, 1.05)
            
            # Set x-axis based on available purification levels
            if not df_p.empty:
                max_ell = df_p['purification_level'].max()
                min_ell = df_p['purification_level'].min()
                ax.set_xlim(min_ell - 0.1, max_ell + 0.1)
                # Set reasonable tick marks
                if max_ell <= 5:
                    ax.set_xticks(range(int(min_ell), int(max_ell) + 1))
                else:
                    ax.set_xticks(range(int(min_ell), int(max_ell) + 1, 2))
            
            # Add legend to top-left subplot
            if plot_idx == 0:
                ax.legend(fontsize=20, loc='lower right')
                
            if plot_idx == 1 and noise_type == 'theta_phi_dephasing':
                ax.axhline(1.0, color='gray', linestyle='dotted', linewidth=5.0, alpha=1.0)
                
            # Y-axis label only on first column
            if plot_idx == 0 or plot_idx == 2:
                ax.set_ylabel(r'Fidelity, $F$', fontsize=40)
                
            # X-axis label only on bottom row
            if plot_idx == 2 or plot_idx == 3:
                ax.set_xlabel(r'Purification Rounds, $\ell$', fontsize=40)
                
            # Add subplot label with same positioning as original
            if plot_idx == 0 or plot_idx == 1:
                ax.text(0.08, 0.02, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='bottom', ha='right')
            else:
                ax.text(0.08, 0.88, subplot_labels[plot_idx], transform=ax.transAxes, fontsize=28, 
                    fontweight='bold', fontfamily='sans-serif', va='bottom', ha='right')

        # Hide unused subplots (if less than 4 p values found)
        for plot_idx in range(len(ps), 4):
            axes_flat[plot_idx].set_visible(False)

        plt.tight_layout()

        filename = f"fidelity_vs_purification_rounds_iter1_{filename_suffix}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_fidelity_combined_M_mini_4x2(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Combined fidelity plots: 4x2 grid with:
        Row 1: depolarizing
        Row 2: dephasing (untwirled)  
        Row 3: dephasing (approx twirl)
        Row 4: dephasing (full twirl)

        Each column represents M=1 and M=5 only.
        This version plots FIDELITY instead of error rate.
        """
        # Check if we have *any* data
        if (self.depol_steps.empty and self.dephase_untwirled_steps.empty and
            self.dephase_approx_steps.empty and self.dephase_twirled_steps.empty):
            print("No data for fidelity plots")
            return None

        # Create 4x2 subplot grid
        fig, axes = plt.subplots(4, 2, figsize=(12, 20))

        # Noise type configurations (row order matters)
        noise_configs = [
            {
                'type': 'depolarizing',
                'df_data': self.depol_steps,
                'row_idx': 0,
                'row_label': 'Depolarizing',
                'subplot_label': ['a', 'b'],
                'use_finals': False  # Use iteration==1 data
            },
            {
                'type': 'dephasing_untwirled',
                'df_data': self.dephase_untwirled_steps,
                'row_idx': 1,
                'row_label': 'Dephasing (Untwirled)',
                'subplot_label': ['c', 'd'],
                'use_finals': False  # Use iteration==1 data
            },
            {
                'type': 'dephasing_approx',
                'df_data': self.dephase_approx_steps,
                'row_idx': 2,
                'row_label': 'Dephasing (Approx Twirl)',
                'subplot_label': ['e', 'f'],
                'use_finals': False  # Use iteration==1 data
            },
            {
                'type': 'dephasing_fullTwirl',
                'df_data': self.dephase_twirled_steps,
                'row_idx': 3,
                'row_label': 'Dephasing (Full Twirl)',
                'subplot_label': ['g', 'h'],
                'use_finals': False  # Use iteration==1 data
            },
        ]

        # M values to plot - only M=1 and M=5
        M_values = [1, 5]

        for noise_config in noise_configs:
            df = noise_config['df_data']
            row_idx = noise_config['row_idx']
            noise_type = noise_config['type']
            use_finals = noise_config['use_finals']

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

            # For VD: Filter for iteration==1 from steps data, then get final fidelity
            if not use_finals:
                df = df[df['iteration'] == 1].copy()
                
                # Get final fidelity at iteration==1 by taking max depth (and max merge_num if depth ties)
                # Group by (M, purification_level, p) and get the last row after sorting by depth, merge_num
                sort_cols = ['M', 'purification_level', 'p', 'depth']
                if 'merge_num' in df.columns:
                    sort_cols.append('merge_num')
                
                df = df.sort_values(sort_cols)
                df = df.groupby(['M', 'purification_level', 'p'], as_index=False).last()

            # Plot each M value
            for col_idx, M in enumerate(M_values):
                ax = axes[row_idx, col_idx]

                # Filter for this M value
                df_M = df[df['M'] == M].copy()

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
                    # For steps data - use purification_level
                    N_values = sorted(df_M['purification_level'].unique()) if 'purification_level' in df_M.columns else sorted(df_M['N'].unique())
                    p_col = 'p' if 'p' in df_M.columns else 'p_channel'
                    fidelity_col = 'fidelity'

                    colors = [
                        'red', 'green', 'blue', 'orange', 'purple', 'saddlebrown',
                        'deeppink', 'darkslategrey', 'fuchsia', 'gold'
                    ]

                    # Plot curves for different purification_level values
                    for i, N in enumerate(N_values):
                        if 'purification_level' in df_M.columns:
                            df_N = df_M[df_M['purification_level'] == N].sort_values(p_col)
                            ell_label = N
                        else:
                            df_N = df_M[df_M['N'] == N].sort_values(p_col)
                            ell_label = int(np.log2(N)) if N > 0 else N

                        if len(df_N) == 0:
                            continue

                        # Use fidelity directly
                        fidelity = df_N[fidelity_col]

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

        filename = f"fidelity_combined_M_4rows_VD.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_combined_fidelity_and_gamma_2x2_grid(self, noise_type: str, save_format: str = 'pdf', 
                                              *, p_center: float = 0.80, p_halfwidth: float = 0.05):
        """
        Combined 2x2 grid plot:
        - Top row (a,b): Fidelity vs iterations for p=0.1, 0.3 (M=5)
        - Bottom row (c,d): Gamma vs p for M=1, M=5 (with inset on M=5)
        
        Args:
            noise_type: 'depolarizing' or 'dephasing'
            save_format: 'pdf', 'png', etc.
            p_center: Center p-value for inset zoom
            p_halfwidth: Half-width for inset zoom window
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.ticker as mticker
        import numpy as np
        
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df_steps = self.depol_steps
            twirling_filter = False
            title_str = "Depolarizing"
        elif noise_type == 'approx_twirled_dephasing':  # approx twirled dephasing
            df_steps = self.dephase_approx_steps
            twirling_filter = True
            title_str = "Approx. Twirled Dephasing"
        else:  # dephasing
            df_steps = self.dephase_twirled_steps
            twirling_filter = True
            title_str = "Dephasing (twirled Z)"

        if df_steps.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Check for required columns
        required_cols = ['iteration', 'purification_level', 'M', 'p', 'fidelity']
        missing_cols = [col for col in required_cols if col not in df_steps.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None

        # Handle twirling column name
        twirling_col = 'twirling_applied' if 'twirling_applied' in df_steps.columns else 'twirling_enabled'
        if twirling_col not in df_steps.columns:
            print(f"No twirling column found in {noise_type} data")
            return None

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 
                'deeppink', 'darkslategrey', 'fuchsia', 'gold']
        
        # Subplot labels
        subplot_labels = [['a', 'b'], ['c', 'd']]
        
        # ================================================================
        # TOP ROW: FIDELITY VS ITERATIONS (M=5, p=0.1 and p=0.3)
        # ================================================================
        
        print("Plotting top row: Fidelity vs Iterations")
        
        # Filter for M=5 and correct twirling
        df_fid_filtered = df_steps[(df_steps['M'] == 5) & (df_steps[twirling_col] == twirling_filter)].copy()
        print(f"Fidelity plots: {len(df_fid_filtered)} rows after M=5, twirling filter")
        
        if not df_fid_filtered.empty:
            l_values = sorted(df_fid_filtered['purification_level'].unique())
            l_to_color_idx = {l_val: idx for idx, l_val in enumerate(l_values)}
            p_subset_fid = [0.1, 0.3]  # For top row
            l_values = [0, 1, 2, 3, 4, 5] 
            
            for i, p_val in enumerate(p_subset_fid):
                ax = axes[0, i]  # Top row
                
                # Find closest p-value
                available_p = df_fid_filtered['p'].unique()
                closest_p = min(available_p, key=lambda x: abs(x - p_val))
                
                df_p = df_fid_filtered[abs(df_fid_filtered['p'] - closest_p) <= 0.02].copy()
                
                if df_p.empty:
                    ax.text(0.5, 0.5, f'No data\\nfor p = {p_val:.1f}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    continue
                
                # Plot curves for different ℓ values
                for j, l_val in enumerate(l_values):
                    df_l = df_p[df_p['purification_level'] == l_val].copy()
                    
                    if df_l.empty:
                        continue
                    
                    # Extract final fidelity for each iteration (max depth/merge_num)
                    sort_cols = ['iteration', 'depth']
                    if 'merge_num' in df_l.columns:
                        sort_cols.append('merge_num')
                    
                    df_l = df_l.sort_values(sort_cols)
                    df_l = df_l.groupby(['iteration'], as_index=False).last()
                    
                    # Sort and get arrays
                    df_l = df_l.sort_values('iteration')
                    x = df_l['iteration'].to_numpy()
                    y = df_l['fidelity'].to_numpy()
                    
                    if len(x) == 0:
                        continue
                    
                    # Prepend (0,1) if needed
                    if x[0] != 0:
                        x = np.insert(x, 0, 0)
                        y = np.insert(y, 0, 1.0)
                    cidx = l_to_color_idx[l_val]
                    # Plot
                    if l_val == 0:
                        ax.plot(x, y, linestyle='dotted', marker=_mk(cidx), 
                            color=colors[cidx % len(colors)], linewidth=2, markersize=12, 
                            alpha=0.8, label=rf'No QEC')
                    else:
                        ax.plot(x, y, linestyle='-', marker=_mk(cidx), 
                            color=colors[cidx % len(colors)], linewidth=2, markersize=12, 
                            alpha=0.8, label=rf'$\ell$ = {l_val}')
                
                # Formatting for fidelity plots
                ax.set_title(f'p = {closest_p:.1f}', fontsize=55)
                ax.set_ylim(0.005, 1.05)
                ax.set_yscale('log')
                
                # X-axis
                max_iter = max(10, df_p['iteration'].max()) if len(df_p) > 0 else 10
                ax.set_xlim(0.0, max_iter + 0.5)
                ax.set_xticks(range(0, int(max_iter) + 1, 2))
                ax.set_xlabel(r'PQEC Cycles, $t$', fontsize=50)
                
                # Y-axis label only on left
                if i == 0:
                    ax.set_ylabel(r'Fidelity, $F$', fontsize=50)
                
                # Legend on right plot
                if i == 0:
                    ax.legend(fontsize=28, loc='best', frameon=False, ncol=2)
                
                # Subplot label
                ax.text(0.08, 0.14, subplot_labels[0][i], transform=ax.transAxes, 
                    fontsize=50, fontweight='bold', va='top', ha='right')
        
        # ================================================================
        # BOTTOM ROW: GAMMA VS P (M=1 and M=5)
        # ================================================================
        
        print("Plotting bottom row: Gamma vs p")
        
        # Prepare gamma data
        df_tw = df_steps[df_steps[twirling_col] == twirling_filter].copy()
        df_t1 = df_tw[df_tw['iteration'] == 1].copy()
        
        if not df_t1.empty:
            # Extract final fidelity at iteration==1 (max depth/merge_num)
            sort_cols = ['M', 'purification_level', 'p', 'depth']
            if 'merge_num' in df_t1.columns:
                sort_cols.append('merge_num')
            
            df_t1 = df_t1.sort_values(sort_cols)
            df_t1_final = df_t1.groupby(['M', 'purification_level', 'p'], as_index=False).last()
            
            # Calculate gamma = 1 - F(t=1)
            df_gamma = df_t1_final[['M', 'purification_level', 'p', 'fidelity']].copy()
            df_gamma = df_gamma.rename(columns={'fidelity': 'F_t1'})
            df_gamma['gamma'] = 1.0 - df_gamma['F_t1']
            
            l_values_gamma = sorted(df_gamma['purification_level'].unique())
            l_to_color_idx = {l_val: idx for idx, l_val in enumerate(l_values_gamma)}
            l_values_gamma = [0, 1, 2, 3, 4, 5] 
            
            # Exclusion set for main plots
            exclude = {0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79}
            
            # Inset zoom parameters
            p_min_zoom = max(0.0, p_center - p_halfwidth)
            p_max_zoom = min(1.0, p_center + p_halfwidth)
            inset_ylim = (0.98, 0.99)  # For M=5 inset
            
            target_Ms = [1, 5]
            
            for j, M in enumerate(target_Ms):
                ax = axes[1, j]  # Bottom row
                
                df_M = df_gamma[df_gamma['M'] == M].copy()
                
                # Formatting
                ax.set_title(f'M={M}', fontsize=55)
                ax.set_xlabel('Physical Error Rate, p', fontsize=50)
                if j == 0:  # Left plot only
                    ax.set_ylabel(r'Logical Error, $\gamma_L$', fontsize=50)
                
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.05)
                
                if df_M.empty:
                    ax.text(0.5, 0.5, f'No data for M={M}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                    continue
                
                # Plot main gamma curves
                handles, labels = [], []
                for l_val in l_values_gamma:
                    df_ml = df_M[df_M['purification_level'] == l_val].copy()
                    if df_ml.empty:
                        continue
                    
                    # Apply exclusion filter
                    p_round = df_ml['p'].round(2)
                    df_main = df_ml[~p_round.isin(exclude)].copy()
                    
                    if df_main.empty:
                        continue
                    
                    # Sort and plot
                    df_main = df_main.sort_values('p')
                    linestyle = ':' if int(l_val) == 0 else '-'
                    cidx = l_to_color_idx[l_val]
                    
                    if l_val == 0:
                        label = rf'No QEC'
                    else:
                        label = rf'$\ell={l_val}$'
                    line, = ax.plot(df_main['p'], df_main['gamma'],
                                linestyle=linestyle, marker=_mk(cidx),
                                linewidth=2.5, markersize=12, alpha=0.85,
                                color=colors[cidx % len(colors)],
                                label=label)
                    handles.append(line)
                    labels.append(label)
                
                # Legend only on M=1 (left plot)
                if M == 1:
                    ax.legend(handles, labels, fontsize=28, loc='best', frameon=False, ncol=1)
                
                # Subplot label
                ax.text(0.08, 0.98, subplot_labels[1][j], transform=ax.transAxes,
                    fontsize=50, fontweight='bold', va='top', ha='right')
                
                # ========================
                # INSET ONLY FOR M=5
                # ========================
                # if M == 5:
                #     axins = inset_axes(ax, width='48%', height='42%', 
                #                     loc='lower right', borderpad=1.1)
                    
                #     # Plot full curves in inset
                #     for l_val in l_values_gamma:
                #         df_ml = df_M[df_M['purification_level'] == l_val].copy()
                #         if df_ml.empty:
                #             continue
                        
                #         df_ml = df_ml.sort_values('p')
                #         linestyle = ':' if int(l_val) == 0 else '-'
                #         cidx = l_to_color_idx[l_val]
                        
                #         x = df_ml['p'].to_numpy(dtype=float)
                #         y = df_ml['gamma'].to_numpy(dtype=float)
                        
                #         # Filter positive values for log scale
                #         mask = y > 0.0
                #         if mask.sum() < 2:
                #             continue
                        
                #         axins.plot(x[mask], y[mask], linestyle=linestyle, marker=_mk(cidx),
                #                 linewidth=2.0, markersize=6, alpha=0.9,
                #                 color=colors[cidx % len(colors)])
                    
                #     # Inset formatting
                #     axins.set_xlim(p_min_zoom, p_max_zoom)
                #     axins.set_yscale('log')
                    
                #     # Set y-limits for crossover visibility
                #     ymin, ymax = inset_ylim
                #     ymin = max(float(ymin), 1e-12)
                #     ymax = max(float(ymax), ymin * 1.01)
                #     axins.set_ylim(ymin, ymax)
                    
                #     # Inset tick formatting
                #     axins.tick_params(axis='both', which='major', labelsize=15)
                #     axins.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                #     axins.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
                #     axins.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
                #     axins.yaxis.set_minor_formatter(mticker.NullFormatter())
        
        # Overall formatting
        plt.tight_layout()
        
        # Save
        filename = f"combined_fidelity_gamma_2x2_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)
    
    def generate_all_plots(self, save_format: str = 'pdf') -> Dict[str, Optional[str]]:
        """Generate all VD figures."""
        print("\n" + "="*70)
        print("GENERATING VD FIGURES")
        print("="*70)
        
        plots = {}
        
        # Plot 1: Fidelity vs purification rounds (iteration==1, 2x2 grid)
        print("\n1. Fidelity vs purification rounds (iteration==1)...")
        plots['fidelity_vs_ell_iter1_depol'] = self.plot_fidelity_vs_purification_rounds_iteration1('depolarizing', save_format)
        plots['fidelity_vs_ell_iter1_dephase'] = self.plot_fidelity_vs_purification_rounds_iteration1('untwirled_dephasing', save_format)
        plots['fidelity_vs_ell_iter1_dephase_approx'] = self.plot_fidelity_vs_purification_rounds_iteration1('approx_twirled_dephasing', save_format)
        plots['fidelity_vs_ell_iter1_dephase_twirled'] = self.plot_fidelity_vs_purification_rounds_iteration1('theta_phi_dephasing', save_format)
        
        # Plot 2: Combined fidelity (4x2 grid: all noise types vs M=1,5)
        print("\n2. Combined fidelity plot (4x2 grid)...")
        plots['fidelity_combined_4x2'] = self.plot_fidelity_combined_M_mini_4x2(save_format)
        
        # Plot 3: Combined fidelity and gamma (2x2 grid)
        print("\n3. Combined fidelity and gamma plot (2x2 grid)...")
        plots['combined_fidelity_gamma_depol'] = self.plot_combined_fidelity_and_gamma_2x2_grid('depolarizing', save_format)
        plots['combined_fidelity_gamma_dephase'] = self.plot_combined_fidelity_and_gamma_2x2_grid('approx_twirled_dephasing', save_format)
        
        # Summary
        successful = [name for name, path in plots.items() if path is not None]
        print(f"\n{len(successful)}/{len(plots)} plots generated successfully")
        print(f"Figures saved to: {self.figures_dir}")
        
        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        
        return plots


def main():
    """Main function."""
    import sys
    
    data_dir = "data/VD_sim"
    figures_dir = "figures/VD_results"
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
    
    plotter = VirtualDistillationPlotter(data_dir, figures_dir)
    plots = plotter.generate_all_plots(save_format)
    
    return plots


if __name__ == "__main__":
    main()