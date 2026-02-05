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
        self.dephase_untwirled_finals = self._load_csv('finals_circuit_dephasing_v4.csv')  # v4 for no twirling
        self.dephase_untwirled_steps = self._load_csv('steps_circuit_dephasing_v4.csv')
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
                ax.legend(fontsize=20, loc='lower right')
                
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
                    
                    # Add legend to one subplot (bottom right)
                    if row_idx == 2 and col_idx == 1 and len(N_values) > 0:
                        ax.legend(fontsize=16, loc='lower right', frameon=False)

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
        
        print("\n2. Combined fidelity plot (3x2 grid: all noise types vs M=1,5)...")
        plots['fidelity_combined_M_3rows'] = self.plot_fidelity_combined_M_mini(save_format)
        
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