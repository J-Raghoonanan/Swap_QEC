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


class SimulationPlotter:
    """Generate figures from simulation CSV data."""
    
    def __init__(self, data_dir: str = "data/simulations", 
                 figures_dir: str = "figures/results_v3_sim"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.depol_finals = self._load_csv('finals_circuit_depolarizing.csv')
        self.depol_steps = self._load_csv('steps_circuit_depolarizing.csv')
        self.dephase_finals = self._load_csv('finals_circuit_dephasing.csv')
        self.dephase_steps = self._load_csv('steps_circuit_dephasing.csv')
        
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
                    # Format: M1_N512_dephase_z_iid_p_d0.50000_twirl
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
            param_label = r'$\delta$'  # Use delta for depolarizing
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
            df_N = df_m1[df_m1['N'] == N].sort_values('delta')
        
            if len(df_N) > 0:
                ax.semilogy(df_N['delta'], df_N['eps_L_final'], 'o-',
                        color=colors[i], linewidth=3, markersize=8,
                        label=f'N = {N}', alpha=0.8)
    
        # No correction reference
        delta_range = np.logspace(-2, 0, 100)
        ax.semilogy(delta_range, delta_range, '--',
                color='gray', linewidth=2, alpha=0.7, label='No Correction')
    
        ax.set_xlabel(f'Physical Error Rate, {param_label}', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'QEC Threshold\n({title_str} Noise, M=1)', fontsize=30)
    
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
        Error evolution for M=1: Error vs purification depth for different delta.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            color = COLORS['depolarizing']
            twirling_filter = False
            param_symbol = r'\delta'
        else:
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
            param_symbol = 'p'
    
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
    
        # Get unique deltas
        deltas = sorted(df_N['delta'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(deltas)))
    
        for i, delta in enumerate(deltas):
            df_delta = df_N[df_N['delta'] == delta].copy()
        
            # Group by depth and take best (min) error
            evolution = df_delta.groupby('depth')['eps_L'].min().reset_index()
        
            if len(evolution) > 0:
                ax.semilogy(evolution['depth'], evolution['eps_L'], 'o-',
                        color=colors[i], linewidth=3, markersize=6,
                        label=f'${param_symbol}={delta:.2f}$', alpha=0.8)
    
        ax.set_xlabel(r'Purification Level, $n$', fontsize=25)
        ax.set_ylabel(r'Logical Error Rate, $\varepsilon_L^{(n)}$', fontsize=25)
    
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
        Fidelity evolution for M=1: Fidelity vs purification depth for different delta.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            twirling_filter = False
            param_symbol = r'\delta'
        else:
            df = self.dephase_steps
            twirling_filter = True
            param_symbol = 'p'
    
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
    
        # Get unique deltas
        deltas = sorted(df_N['delta'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(deltas)))
    
        for i, delta in enumerate(deltas):
            df_delta = df_N[df_N['delta'] == delta].copy()
        
            # Group by depth and take best (max) fidelity
            evolution = df_delta.groupby('depth')['fidelity'].max().reset_index()
        
            if len(evolution) > 0:
                ax.plot(evolution['depth'], evolution['fidelity'], 'o-',
                    color=colors[i], linewidth=3, markersize=8,
                    label=f'${param_symbol}={delta:.2f}$')
    
        # Target fidelity line
        ax.axhline(y=0.99, color='black', linestyle=':', alpha=0.7,
                linewidth=2, label='Target 0.99')
    
        ax.set_xlabel('Purification Level', fontsize=25)
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
        NEW: Final error vs delta for different M values at fixed max N.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_finals
            color = COLORS['depolarizing']
            twirling_filter = False
            param_label = r'$\delta$'
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
            df_M = df_N[df_N['M'] == M].sort_values('delta')
        
            if len(df_M) > 0:
                ax.semilogy(df_M['delta'], df_M['eps_L_final'], 'o-',
                        color=colors[i], linewidth=3, markersize=8,
                        label=f'M = {M}', alpha=0.85)
    
        # No correction reference
        delta_range = np.logspace(-2, 0, 100)
        ax.semilogy(delta_range, delta_range, '--',
                color='gray', linewidth=2, alpha=0.7, label='No Correction')
    
        ax.set_xlabel(f'Physical Error Rate, {param_label}', fontsize=25)
        ax.set_ylabel(r'Final Logical Error Rate, $\varepsilon_L$', fontsize=25)
    
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
        NEW: Final fidelity vs M for different delta values at fixed max N.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_finals
            twirling_filter = False
            param_symbol = r'\delta'
        else:
            df = self.dephase_finals
            twirling_filter = True
            param_symbol = 'p'
    
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
    
        # Get unique deltas (select a representative subset)
        all_deltas = sorted(df_N['delta'].unique())
        # Choose ~5-6 well-spaced deltas
        if len(all_deltas) > 6:
            step = len(all_deltas) // 6
            deltas = all_deltas[::step]
        else:
            deltas = all_deltas
    
        colors = plt.cm.viridis(np.linspace(0, 1, len(deltas)))
    
        for i, delta in enumerate(deltas):
            df_delta = df_N[df_N['delta'] == delta].sort_values('M')
        
            if len(df_delta) > 0:
                ax.plot(df_delta['M'], df_delta['fidelity_final'], 'o-',
                    color=colors[i], linewidth=3, markersize=8,
                    label=f'${param_symbol}={delta:.2f}$')
    
        # Target fidelity line
        ax.axhline(y=0.99, color='black', linestyle=':', alpha=0.7,
                linewidth=2, label='Target 0.99')
    
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
    
    def generate_all_plots(self, save_format: str = 'pdf') -> Dict[str, Optional[str]]:
        """Generate all figures."""
        print("\n" + "="*70)
        print("GENERATING SIMULATION FIGURES")
        print("="*70)
        
        plots = {}
        
        # M=1 threshold plots
        print("\n1. Threshold plots (M=1)...")
        plots['threshold_depol_m1'] = self.plot_threshold_m1('depolarizing', save_format)
        plots['threshold_dephase_m1'] = self.plot_threshold_m1('dephasing', save_format)
        
        # M=1 error evolution
        print("\n2. Error evolution (M=1)...")
        plots['error_evol_depol_m1'] = self.plot_error_evolution_m1('depolarizing', save_format)
        plots['error_evol_dephase_m1'] = self.plot_error_evolution_m1('dephasing', save_format)
        
        # M=1 fidelity evolution
        print("\n3. Fidelity evolution (M=1)...")
        plots['fidelity_evol_depol_m1'] = self.plot_fidelity_evolution_m1('depolarizing', save_format)
        plots['fidelity_evol_dephase_m1'] = self.plot_fidelity_evolution_m1('dephasing', save_format)
        
        # NEW: Multi-M threshold
        print("\n4. Threshold vs M (max N)...")
        plots['threshold_vs_M_depol'] = self.plot_threshold_vs_M('depolarizing', save_format)
        plots['threshold_vs_M_dephase'] = self.plot_threshold_vs_M('dephasing', save_format)
        
        # NEW: Fidelity vs M
        print("\n5. Fidelity vs M (max N)...")
        plots['fidelity_vs_M_depol'] = self.plot_fidelity_vs_M('depolarizing', save_format)
        plots['fidelity_vs_M_dephase'] = self.plot_fidelity_vs_M('dephasing', save_format)
        
        # Summary
        successful = [name for name, path in plots.items() if path is not None]
        print(f"\n{len(successful)}/{len(plots)} plots generated successfully")
        print(f"Figures saved to: {self.figures_dir}")
        
        return plots


def main():
    """Main function."""
    import sys
    
    data_dir = "data/simulations"
    figures_dir = "figures/results_v3_sim"
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