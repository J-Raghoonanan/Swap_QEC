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
    'axes.titlesize': 30,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 30,
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 3,
    'lines.markersize': 8
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
        self.dephase_finals = self._load_csv('finals_circuit_dephasing_v2.csv')
        self.dephase_steps = self._load_csv('steps_circuit_dephasing_v2.csv')
        
        print(f"Loaded simulation data:")
        print(f"  Depolarizing finals: {len(self.depol_finals)} runs")
        print(f"  Depolarizing steps: {len(self.depol_steps)} merges")
        print(f"  Dephasing finals: {len(self.dephase_finals)} runs")
        print(f"  Dephasing steps: {len(self.dephase_steps)} merges")
    
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
    
    def plot_threshold_m1(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Threshold plot for M=1: Final error vs physical error rate for different N.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_finals
            color = COLORS['depolarizing']
            twirling_filter = False
            param_label = r'$p$'  # Now use p for depolarizing
        else:  # dephasing
            df = self.dephase_finals
            color = COLORS['dephasing']
            twirling_filter = True
            param_label = r'$p$'  # Use p for dephasing
    
        if df.empty:
            print(f"No data for {noise_type} threshold plot")
            return None
    
        # Filter M=1 AND twirling condition
        df_m1 = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_m1.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique N values
        N_values = sorted(df_m1['N'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
    
        for i, N in enumerate(N_values):
            df_N = df_m1[df_m1['N'] == N].sort_values('p_channel')  # Changed from 'delta' to 'p'
        
            if len(df_N) > 0:
                ax.semilogy(df_N['p_channel'], df_N['eps_L_final'],  # Changed from 'delta' to 'p'
                            linestyle='-', marker=_mk(i),
                        color=colors[i], linewidth=3, markersize=8,
                        label=f'N = {N}', alpha=0.8)
    
        # No correction reference
        p_range = np.logspace(-2, 0, 100)  # Changed from delta_range to p_range
        ax.semilogy(p_range, p_range, '--',
                color='gray', linewidth=2, alpha=0.7, label='No Correction')
    
        ax.set_xlabel(f'Physical Error Rate, {param_label}', fontsize=25)
        ax.set_ylabel(r'Final Error Rate, $\varepsilon$', fontsize=25)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'PEC Threshold\n({title_str} Noise, M=1)', fontsize=30)
    
        ax.legend(fontsize=14, loc='lower right')
        ax.set_xlim(0.09, 1.0)
        ax.set_ylim(1e-5, 1.0)
    
        plt.tight_layout()
    
        filename = f"threshold_{noise_type}_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)

    
    def plot_error_evolution_m1(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Error evolution for M=1: Error vs purification depth for different p.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            color = COLORS['depolarizing']
            twirling_filter = False
            param_symbol = r'p'  # Changed from r'\delta' to r'p'
        else:
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
            param_symbol = r'p'
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Filter M=1 AND twirling condition
        df_m1 = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_m1.empty:
            print(f"No M=1 steps for {noise_type} with twirling={twirling_filter}")
            return None
    
        # Use max N
        max_N = df_m1['N'].max()
        df_N = df_m1[df_m1['N'] == max_N].copy()
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique p values (changed from deltas)
        ps = sorted(df_N['p_channel'].unique())  # Changed from 'delta' to 'p'
        colors = plt.cm.viridis(np.linspace(0, 1, len(ps)))
    
        for i, p in enumerate(ps):  # Changed from delta to p
            df_p = df_N[df_N['p_channel'] == p].copy()  # Changed from 'delta' to 'p'
        
            # Group by depth and take best (min) error
            evolution = df_p.groupby('depth')['eps_L'].min().reset_index()
        
            if len(evolution) > 0:
                ax.semilogy(evolution['depth'], evolution['eps_L'],
                            linestyle='-', marker=_mk(i),
                        color=colors[i], linewidth=3, markersize=6,
                        label=f'${param_symbol}={p:.2f}$', alpha=0.8)  # Changed from delta to p
    
        ax.set_xlabel(r'Rounds of Purification, $n$', fontsize=25)
        ax.set_ylabel(r'Error Rate, $\varepsilon^{(n)}$', fontsize=25)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'Error Evolution\n({title_str}, M=1, N={max_N})', fontsize=30)
    
        ax.legend(fontsize=14, loc='best')

        plt.tight_layout()
    
        filename = f"error_evolution_{noise_type}_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_fidelity_evolution_m1(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Fidelity evolution for M=1: Fidelity vs purification depth for different p.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            twirling_filter = False
            param_symbol = r'p'  # Changed from r'\delta' to r'p'
        else:
            df = self.dephase_steps
            twirling_filter = True
            param_symbol = r'p'
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Filter M=1 AND twirling condition
        df_m1 = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_m1.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        # Use max N
        max_N = df_m1['N'].max()
        df_N = df_m1[df_m1['N'] == max_N].copy()
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique p values (changed from deltas)
        ps = sorted(df_N['p_channel'].unique())  # Changed from 'delta' to 'p'
        colors = plt.cm.viridis(np.linspace(0, 1, len(ps)))
    
        for i, p in enumerate(ps):  # Changed from delta to p
            df_p = df_N[df_N['p_channel'] == p].copy()  # Changed from 'delta' to 'p'

            # Group by depth and take best (max) fidelity
            evolution = df_p.groupby('depth')['fidelity'].max().reset_index()
        
            if len(evolution) > 0:
                ax.plot(evolution['depth'], evolution['fidelity'],
                        linestyle='-', marker=_mk(i),
                        color=colors[i], linewidth=3, markersize=8,
                        label=f'${param_symbol}={p:.2f}$')  # Changed from delta to p
    
        # Target fidelity line
        # ax.axhline(y=0.99, color='black', linestyle=':', alpha=0.7,
        #         linewidth=2, label='Target 0.99')
    
        ax.set_xlabel('Rounds of Purification', fontsize=25)
        ax.set_ylabel('State Fidelity', fontsize=25)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'Fidelity Evolution\n({title_str}, M=1, N={max_N})', fontsize=30)
    
        ax.legend(fontsize=14, loc='best')
        ax.set_ylim(0, 1.05)
    
        plt.tight_layout()
    
        filename = f"fidelity_evolution_{noise_type}_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)
    
    def plot_threshold_vs_M(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        NEW: Final error vs p for different M values at fixed max N.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_finals
            color = COLORS['depolarizing']
            twirling_filter = False
            param_label = r'$p$'  # Changed from r'$\delta$' to r'$p$'
        else:
            df = self.dephase_finals
            color = COLORS['dephasing']
            twirling_filter = True
            param_label = r'$p$'
    
        if df.empty:
            print(f"No data for {noise_type}")
            return None
    
        # Filter by twirling condition
        df = df[df['twirling_enabled'] == twirling_filter].copy()
    
        if df.empty:
            print(f"No data for {noise_type} with twirling={twirling_filter}")
            return None
    
        # Use max N
        max_N = df['N'].max()
        df_N = df[df['N'] == max_N].copy()
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique M values
        M_values = sorted(df_N['M'].unique())
        colors = plt.cm.plasma(np.linspace(0, 1, len(M_values)))
    
        for i, M in enumerate(M_values):
            df_M = df_N[df_N['M'] == M].sort_values('p_channel')  # Changed from 'delta' to 'p'
        
            if len(df_M) > 0:
                ax.semilogy(df_M['p_channel'], df_M['eps_L_final'],  # Changed from 'delta' to 'p'
                            linestyle='-', marker=_mk(i),
                            color=colors[i], linewidth=3, markersize=8,
                            label=f'M = {M}', alpha=0.85)
    
        # No correction reference
        p_range = np.logspace(-2, 0, 100)  # Changed from delta_range to p_range
        ax.semilogy(p_range, p_range, '--',
                color='gray', linewidth=2, alpha=0.7, label='No Correction')
    
        ax.set_xlabel(f'Physical Error Rate, {param_label}', fontsize=25)
        ax.set_ylabel(r'Final Error Rate, $\varepsilon$', fontsize=25)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'System Size Scaling\n({title_str}, N={max_N})', fontsize=30)
    
        ax.legend(fontsize=14, loc='lower right')
        ax.set_xlim(0.09, 1.0)
        ax.set_ylim(1e-5, 1.0)
    
        plt.tight_layout()
    
        filename = f"threshold_vs_M_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)

    def plot_fidelity_vs_M(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        NEW: Final fidelity vs M for different p values at fixed max N.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_finals
            twirling_filter = False
            param_symbol = r'p'  # Changed from r'\delta' to r'p'
        else:
            df = self.dephase_finals
            twirling_filter = True
            param_symbol = r'p'
    
        if df.empty:
            print(f"No data for {noise_type}")
            return None
    
        # Filter by twirling condition
        df = df[df['twirling_enabled'] == twirling_filter].copy()
    
        if df.empty:
            print(f"No data for {noise_type} with twirling={twirling_filter}")
            return None
    
        # Use max N
        max_N = df['N'].max()
        df_N = df[df['N'] == max_N].copy()
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique p values (changed from deltas)
        all_ps = sorted(df_N['p_channel'].unique())  # Changed from 'delta' to 'p'
        # Choose ~5-6 well-spaced p values
        # if len(all_ps) > 6:
        #     step = len(all_ps) // 6
        #     ps = all_ps[::step]
        # else:
        #     ps = all_ps
        
        ps = all_ps
    
        colors = plt.cm.viridis(np.linspace(0, 1, len(ps)))
    
        for i, p in enumerate(ps):  # Changed from delta to p
            df_p = df_N[df_N['p_channel'] == p].sort_values('M')  # Changed from 'delta' to 'p'
        
            if len(df_p) > 0:
                ax.plot(df_p['M'], df_p['fidelity_final'],
                        linestyle='-', marker=_mk(i),
                        color=colors[i], linewidth=3, markersize=8,
                        label=f'${param_symbol}={p:.2f}$')  # Changed from delta to p
    
        # Target fidelity line©
        # ax.axhline(y=0.99, color='black', linestyle=':', alpha=0.7,
        #         linewidth=2, label='Target 0.99')
    
        ax.set_xlabel('System Size (M qubits)', fontsize=25)
        ax.set_ylabel('Final Fidelity', fontsize=25)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'Fidelity vs System Size\n({title_str}, N={max_N})', fontsize=30)
    
        ax.legend(fontsize=14, loc='best')
        ax.set_ylim(0, 1.05)
    
        plt.tight_layout()
    
        filename = f"fidelity_vs_M_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)
    
    
    def plot_fidelity_grid_vs_depth(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        3x3 grid showing fidelity vs purification level for different p values.
        Each subplot shows curves for different M values at fixed p.
        Skips p=0.01 and uses remaining p values.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            twirling_filter = False
            param_symbol = r'p'
        else:
            df = self.dephase_steps
            twirling_filter = True
            param_symbol = r'p'

        if df.empty:
            print(f"No steps data for {noise_type}")
            return None

        # Filter by twirling condition
        df = df[df['twirling_enabled'] == twirling_filter].copy()

        if df.empty:
            print(f"No data for {noise_type} with twirling={twirling_filter}")
            return None

        # Use max N
        max_N = df['N'].max()
        df_N = df[df['N'] == max_N].copy()

        # Get unique p values, skip p=0.01 and 0.75
        all_ps = sorted([p for p in df_N['p_channel'].unique() if p not in [0.01,0.75]])
        
        if len(all_ps) == 0:
            print(f"No valid p values found for {noise_type} (after skipping p=0.01 and 0.75)")
            return None
        
        # Take up to 9 p values for 3x3 grid
        if len(all_ps) > 9:
            # Select evenly spaced values
            step = len(all_ps) // 9
            ps = all_ps[::step][:9]
        else:
            ps = all_ps

        # Get unique M values
        M_values = sorted(df_N['M'].unique())
        colors = plt.cm.plasma(np.linspace(0, 1, len(M_values)))

        # Create 3x3 subplot grid
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Fidelity Evolution Grid ({noise_type.title()})', 
                    fontsize=30, y=0.96)

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        for plot_idx, p in enumerate(ps):
            if plot_idx >= 9:  # Safety check
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
                            color=colors[i], linewidth=2, markersize=5,
                            label=f'M = {M}', alpha=0.8)

            # Subplot formatting
            ax.set_title(f'${param_symbol} = {p:.2f}$', fontsize=20)
            ax.set_xlabel('Rounds of Purification', fontsize=16)
            ax.set_ylabel('Fidelity', fontsize=16)
            ax.set_ylim(0, 1.05)
            # ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if plot_idx == 0:
                ax.legend(fontsize=12, loc='best')

        # Hide unused subplots
        for plot_idx in range(len(ps), 9):
            axes_flat[plot_idx].set_visible(False)

        plt.tight_layout()

        filename = f"fidelity_grid_vs_depth_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    
    def plot_threshold_multi_M(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Threshold plots for M=1 through M=5: Final error vs physical error rate for different N.
        Creates 5 subplots, one for each M value.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_finals
            color = COLORS['depolarizing']
            twirling_filter = False
            param_label = r'$p$'
        else:  # dephasing
            df = self.dephase_finals
            color = COLORS['dephasing']
            twirling_filter = True
            param_label = r'$p$'

        if df.empty:
            print(f"No data for {noise_type} threshold plot")
            return None

        # Filter by twirling condition
        df_filtered = df[df['twirling_enabled'] == twirling_filter].copy()

        if df_filtered.empty:
            print(f"No data for {noise_type} with twirling={twirling_filter}")
            return None

        # Create 1x5 subplot grid
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        fig.suptitle(f'PEC Threshold vs System Size\n({title_str} Noise)', fontsize=32, y=1.02)

        # M values to plot
        M_values = [1, 2, 3, 4, 5]

        for subplot_idx, M in enumerate(M_values):
            ax = axes[subplot_idx]
            
            # Filter for this M value
            df_M = df_filtered[df_filtered['M'] == M].copy()
            
            if df_M.empty:
                # If no data for this M, show empty subplot with message
                ax.text(0.5, 0.5, f'No data\nfor M={M}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16, alpha=0.7)
                ax.set_title(f'M = {M}', fontsize=20)
                continue

            # Get unique N values for this M
            N_values = sorted(df_M['N'].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))

            # Plot curves for different N values
            for i, N in enumerate(N_values):
                df_N = df_M[df_M['N'] == N].sort_values('p_channel')
                
                if len(df_N) > 0:
                    ax.semilogy(df_N['p_channel'], df_N['eps_L_final'],
                            linestyle='-', marker=_mk(i),
                            color=colors[i], linewidth=2, markersize=6,
                            label=f'N = {N}', alpha=0.8)

            # No correction reference line
            p_range = np.logspace(-2, 0, 100)
            ax.semilogy(p_range, p_range, '--',
                    color='gray', linewidth=1.5, alpha=0.7, label='No Correction')

            # Subplot formatting
            ax.set_title(f'M = {M}', fontsize=20)
            ax.set_xlabel(f'Physical Error Rate, {param_label}', fontsize=16)
            if subplot_idx == 0:  # Only label y-axis on first subplot
                ax.set_ylabel(r'Final Error Rate, $\varepsilon$', fontsize=16)
            
            ax.set_xlim(0.09, 1.0)
            ax.set_ylim(1e-5, 1.0)
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot to avoid clutter
            if subplot_idx == 0 and len(N_values) > 0:
                ax.legend(fontsize=12, loc='lower right')

        plt.tight_layout()

        filename = f"threshold_multi_M_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")
        return str(filepath)
    
    
    
    def plot_threshold_combined_M(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Combined threshold plots: 2x5 grid with depolarizing (top) and dephasing (bottom).
        Each column represents M=1 through M=5.
        """
        # Check if we have data for both noise types
        if self.depol_finals.empty and self.dephase_finals.empty:
            print("No data for threshold plots")
            return None

        # Create 2x5 subplot grid
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle('PEC Threshold vs System Size', fontsize=36, y=0.95)

        # Noise type configurations
        noise_configs = [
            {
                'type': 'depolarizing',
                'df': self.depol_finals,
                'twirling_filter': False,
                'row_idx': 0,
                'row_label': 'Depolarizing Noise'
            },
            {
                'type': 'dephasing', 
                'df': self.dephase_finals,
                'twirling_filter': True,
                'row_idx': 1,
                'row_label': 'Dephasing Noise'
            }
        ]

        # M values to plot
        M_values = [1, 2, 3, 4, 5]
        
        for noise_config in noise_configs:
            df = noise_config['df']
            twirling_filter = noise_config['twirling_filter']
            row_idx = noise_config['row_idx']
            noise_type = noise_config['type']
            
            if df.empty:
                # If no data for this noise type, show empty row with message
                for col_idx in range(5):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, f'No {noise_type}\ndata', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, alpha=0.7)
                    if col_idx == 0:
                        ax.set_ylabel(r'Final Error Rate, $\varepsilon$', fontsize=16)
                    if row_idx == 1:  # Bottom row
                        ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=16)
                continue

            # Filter by twirling condition
            df_filtered = df[df['twirling_enabled'] == twirling_filter].copy()
            
            if df_filtered.empty:
                # If no data after filtering, show empty row
                for col_idx in range(5):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, f'No data\n(twirling={twirling_filter})', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, alpha=0.7)
                    if col_idx == 0:
                        ax.set_ylabel(r'Final Error Rate, $\varepsilon$', fontsize=16)
                    if row_idx == 1:  # Bottom row
                        ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=16)
                continue

            # Plot each M value
            for col_idx, M in enumerate(M_values):
                ax = axes[row_idx, col_idx]
                
                # Filter for this M value
                df_M = df_filtered[df_filtered['M'] == M].copy()
                
                if df_M.empty:
                    # If no data for this M, show message
                    ax.text(0.5, 0.5, f'No data\nfor M={M}', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, alpha=0.7)
                else:
                    # Get unique N values for this M
                    N_values = sorted(df_M['N'].unique())
                    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))

                    # Plot curves for different N values
                    for i, N in enumerate(N_values):
                        df_N = df_M[df_M['N'] == N].sort_values('p_channel')
                        
                        if len(df_N) > 0:
                            ax.semilogy(df_N['p_channel'], df_N['eps_L_final'],
                                    linestyle='-', marker=_mk(i),
                                    color=colors[i], linewidth=2, markersize=6,
                                    label=f'N = {N}', alpha=0.8)

                    # No correction reference line
                    p_range = np.logspace(-2, 0, 100)
                    ax.semilogy(p_range, p_range, '--',
                            color='gray', linewidth=1.5, alpha=0.7, label='No Correction')

                    # Set axis limits and grid
                    ax.set_xlim(0.09, 1.0)
                    ax.set_ylim(1e-5, 1.0)
                    # ax.grid(True, alpha=0.3)
                    
                    # Add legend only to last subplot of second row
                    if row_idx == 1 and col_idx == 4 and len(N_values) > 0:
                        ax.legend(fontsize=11, loc='lower right')

                # Subplot titles (M values) only on top row
                if row_idx == 0:
                    ax.set_title(f'M = {M}', fontsize=20)
                
                # Y-axis label only on first column
                if col_idx == 0:
                    ax.set_ylabel(r'Final Error Rate, $\varepsilon$', fontsize=16)
                
                # X-axis label only on bottom row
                if row_idx == 1:
                    ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=16)

        # Add row labels
        fig.text(0.02, 0.69, 'Depolarizing Noise', rotation=90, fontsize=20, 
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.27, 'Dephasing Noise', rotation=90, fontsize=20, 
                verticalalignment='center', weight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.08)  # Make room for row labels

        filename = f"threshold_combined_M.{save_format}"
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
        
        # # M=1 threshold plots
        # print("\n1. Threshold plots (M=1)...")
        # plots['threshold_depol_m1'] = self.plot_threshold_m1('depolarizing', save_format)
        # plots['threshold_dephase_m1'] = self.plot_threshold_m1('dephasing', save_format)
        
        # # M=1 error evolution
        # print("\n2. Error evolution (M=1)...")
        # plots['error_evol_depol_m1'] = self.plot_error_evolution_m1('depolarizing', save_format)
        # plots['error_evol_dephase_m1'] = self.plot_error_evolution_m1('dephasing', save_format)
        
        # # M=1 fidelity evolution
        # print("\n3. Fidelity evolution (M=1)...")
        # plots['fidelity_evol_depol_m1'] = self.plot_fidelity_evolution_m1('depolarizing', save_format)
        # plots['fidelity_evol_dephase_m1'] = self.plot_fidelity_evolution_m1('dephasing', save_format)
        
        # # NEW: Multi-M threshold
        # print("\n4. Threshold vs M (max N)...")
        # plots['threshold_vs_M_depol'] = self.plot_threshold_vs_M('depolarizing', save_format)
        # plots['threshold_vs_M_dephase'] = self.plot_threshold_vs_M('dephasing', save_format)
        
        # # NEW: Fidelity vs M
        # print("\n5. Fidelity vs M (max N)...")
        # plots['fidelity_vs_M_depol'] = self.plot_fidelity_vs_M('depolarizing', save_format)
        # plots['fidelity_vs_M_dephase'] = self.plot_fidelity_vs_M('dephasing', save_format)
        
        # Fidelity grid vs depth
        # print("\n6. Fidelity grid vs purification level...")
        # plots['fidelity_grid_depol'] = self.plot_fidelity_grid_vs_depth('depolarizing', save_format)
        # plots['fidelity_grid_dephase'] = self.plot_fidelity_grid_vs_depth('dephasing', save_format)
        
        print("\n7. Multi-M threshold plots...")
        # plots['threshold_multi_M_depol'] = self.plot_threshold_multi_M('depolarizing', save_format)
        # plots['threshold_multi_M_dephase'] = self.plot_threshold_multi_M('dephasing', save_format)
        plots['threshold_combined_M'] = self.plot_threshold_combined_M(save_format)

        
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