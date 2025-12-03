"""
Experimental Data Plotter for IBM Quantum Results
Standalone script for plotting experimental data from IBM quantum computers.
Reads data from: data/IBMQ/IBM_results_all_ibm_torino.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Publication-quality plotting settings
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

# Colors for different noise types
COLORS = {
    'depolarizing': '#2E86AB',
    'dephasing': '#F18F01',
}

# Better color palette that avoids hard-to-see yellow
GOOD_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Markers for different series
MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<', 'h', '*']

def _mk(i: int) -> str:
    """Return a distinct marker for series index i (wraps automatically)."""
    return MARKERS[i % len(MARKERS)]


class IBMExperimentalPlotter:
    """Plot experimental quantum error correction results from IBM quantum computers."""
    
    def __init__(self, csv_path: str = "data/IBMQ/IBM_results_all_ibm_torino.csv", 
                 figures_dir: str = "figures/experimental_ibm"):
        """
        Initialize the plotter with IBM experimental data.
        
        Args:
            csv_path: Path to the IBM experimental results CSV
            figures_dir: Directory to save generated figures
        """
        self.csv_path = Path(csv_path)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load experimental data
        self.data = self._load_experimental_data()
        
        if not self.data.empty:
            print(f"Loaded IBM experimental data from: {self.csv_path}")
            print(f"  Total experimental runs: {len(self.data)}")
            print(f"  Noise types: {sorted(self.data['noise_type'].unique())}")
            print(f"  M values: {sorted(self.data['M'].unique())}")
            print(f"  N values: {sorted(self.data['N'].unique())}")
            print(f"  p values: {sorted(self.data['p'].unique())}")
            print(f"  Backend: {self.data['backend_name'].iloc[0] if 'backend_name' in self.data.columns else 'Unknown'}")
        else:
            print(f"Warning: No data loaded from {self.csv_path}")
    
    def _load_experimental_data(self) -> pd.DataFrame:
        """Load the IBM experimental CSV data."""
        if not self.csv_path.exists():
            print(f"Error: CSV file not found at {self.csv_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.csv_path)
            
            # Validate required columns
            required_cols = ['M', 'N', 'p', 'noise_type', 'final_fidelity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Error: Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()

    def plot_threshold_vs_M_combined_experimental(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot both depolarizing and dephasing noise types on the same plot.
        Shows final error rate (1-fidelity) vs physical error rate for different M values.
        Depolarizing: solid lines, Dephasing: dashed lines.
        
        Args:
            save_format: File format for saving ('pdf', 'png', etc.)
            
        Returns:
            Path to saved figure or None if failed
        """
        if self.data.empty:
            print(f"No experimental data available")
            return None
        
        # Use maximum N value available
        max_N = self.data['N'].max()
        df_N = self.data[self.data['N'] == max_N].copy()
        
        if df_N.empty:
            print(f"No data for N={max_N}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique M values and assign colors (using better color palette)
        M_values = sorted(df_N['M'].unique())
        
        # Plot both noise types
        noise_types = ['depolarizing', 'dephasing']
        linestyles = ['-', '--']  # solid for depolarizing, dashed for dephasing
        
        for noise_idx, noise_type in enumerate(noise_types):
            df_noise = df_N[df_N['noise_type'] == noise_type].copy()
            
            if df_noise.empty:
                print(f"No data for {noise_type} noise")
                continue
            
            # Plot each M value for this noise type
            for M_idx, M in enumerate(M_values):
                df_M = df_noise[df_noise['M'] == M].sort_values('p')
                
                if len(df_M) > 0:
                    # Convert fidelity to error rate
                    error_rate = 1 - df_M['final_fidelity']
                    
                    # Use better colors and create label
                    color = GOOD_COLORS[M_idx % len(GOOD_COLORS)]
                    linestyle = linestyles[noise_idx]
                    noise_label = 'Depol.' if noise_type == 'depolarizing' else 'Deph.'
                    label = f'M={M} ({noise_label})'
                    
                    ax.plot(df_M['p'], 1-error_rate,
                               linestyle=linestyle, marker=_mk(M_idx),
                               color=color, linewidth=3, markersize=8,
                               label=label, alpha=0.85)
        
        # Add no-correction reference line
        # p_range = np.logspace(-2, 0, 100)
        # ax.semilogy(p_range, p_range, ':',
        #            color='gray', linewidth=2, alpha=0.7, label='No Correction')
        
        # Formatting
        ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=25)
        ax.set_ylabel('Final Fidelity', fontsize=25)
        # ax.set_title('System Size Scaling', fontsize=28)
        
        ax.legend(fontsize=14, loc='upper right', handlelength=3)
        ax.set_xlim(0.005, 1.0)
        ax.set_ylim(1e-3, 1.0)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"ibm_threshold_vs_M_combined.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
        return str(filepath)

    def plot_fidelity_vs_M_experimental(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot final fidelity vs M for different p values.
        Experimental version of plot_fidelity_vs_M().
        
        Args:
            noise_type: 'depolarizing' or 'dephasing'
            save_format: File format for saving ('pdf', 'png', etc.)
            
        Returns:
            Path to saved figure or None if failed
        """
        if self.data.empty:
            print(f"No experimental data available")
            return None
        
        # Filter by noise type
        df = self.data[self.data['noise_type'] == noise_type].copy()
        if df.empty:
            print(f"No experimental data for {noise_type} noise")
            return None
        
        # Use maximum N value available
        max_N = df['N'].max()
        df_N = df[df['N'] == max_N].copy()
        
        if df_N.empty:
            print(f"No data for N={max_N}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique p values and assign colors
        p_values = sorted(df_N['p'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(p_values)))
        
        # Plot each p value
        for i, p in enumerate(p_values):
            df_p = df_N[df_N['p'] == p].sort_values('M')
            
            if len(df_p) > 0:
                ax.plot(df_p['M'], df_p['final_fidelity'],
                       linestyle='-', marker=_mk(i),
                       color=colors[i], linewidth=3, markersize=8,
                       label=f'$p = {p:.2f}$', alpha=0.85)
        
        # Formatting
        ax.set_xlabel('System Size (M qubits)', fontsize=25)
        ax.set_ylabel('Final Fidelity', fontsize=25)
        
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'Fidelity across System Size\n({title_str} Noise, N={max_N})', fontsize=28)
        
        ax.legend(fontsize=16, loc='best')
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"ibm_fidelity_vs_M_{noise_type}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
        return str(filepath)

    def plot_fidelity_combined_M_experimental(self, save_format: str = 'pdf') -> Optional[str]:
        """
        Combined fidelity plots: 2x2 grid showing fidelity vs p for M=1,2.
        Experimental version with 2x2 grid since only M=1,2 data available.
        
        Args:
            save_format: File format for saving ('pdf', 'png', etc.')
            
        Returns:
            Path to saved figure or None if failed
        """
        if self.data.empty:
            print("No experimental data available")
            return None

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fidelity vs System Size', fontsize=32, y=0.97)

        # Noise types for rows
        noise_types = ['depolarizing', 'dephasing']
        
        # M values for columns (only M=1,2 available)
        M_values = [1, 2]
        
        for row_idx, noise_type in enumerate(noise_types):
            # Filter data for this noise type
            df_noise = self.data[self.data['noise_type'] == noise_type].copy()
            
            if df_noise.empty:
                # If no data for this noise type, show empty row with message
                for col_idx in range(2):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, f'No {noise_type}\ndata', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, alpha=0.7)
                    if col_idx == 0:
                        ax.set_ylabel('Final Fidelity', fontsize=22)
                    if row_idx == 1:  # Bottom row
                        ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=20)
                continue

            # Plot each M value
            for col_idx, M in enumerate(M_values):
                ax = axes[row_idx, col_idx]
                
                # Filter for this M value
                df_M = df_noise[df_noise['M'] == M].copy()
                
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
                        df_N = df_M[df_M['N'] == N].sort_values('p')
                        
                        if len(df_N) > 0:
                            ax.plot(df_N['p'], df_N['final_fidelity'],
                                   linestyle='-', marker=_mk(i),
                                   color=colors[i], linewidth=2, markersize=6,
                                   label=rf'$\ell$ = {int(np.log2(N))}', alpha=0.8)

                    # # No correction reference line (fidelity = 1 - p)
                    # p_range = np.linspace(0.01, 0.9, 100)
                    # fidelity_no_correction = 1 - p_range
                    # ax.plot(p_range, fidelity_no_correction, '--',
                    #        color='gray', linewidth=1.5, alpha=0.7, label='No Correction')

                    # Set axis limits
                    ax.set_xlim(0.005, 1.0)
                    ax.set_ylim(0, 1.05)
                    
                    # Add legend 
                    if row_idx == 0 and col_idx == 0 and len(N_values) > 0:
                        ax.legend(fontsize=12, loc='lower left')

                # Subplot titles (M values) only on top row
                if row_idx == 0:
                    ax.set_title(f'M = {M}', fontsize=26)
                
                # Y-axis label only on first column
                if col_idx == 0:
                    ax.set_ylabel('Final Fidelity', fontsize=22)
                
                # X-axis label only on bottom row
                if row_idx == 1:
                    ax.set_xlabel(r'Physical Error Rate, $p$', fontsize=20)

        # Add row labels
        fig.text(0.02, 0.72, 'Depolarizing Noise', rotation=90, fontsize=24, 
                verticalalignment='center', weight='bold')
        fig.text(0.02, 0.28, 'Dephasing Noise', rotation=90, fontsize=24, 
                verticalalignment='center', weight='bold')

        plt.tight_layout()
        plt.subplots_adjust(left=0.12)  # Make room for row labels

        # Save figure
        filename = f"ibm_fidelity_combined_M.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")
        return str(filepath)

    def generate_all_experimental_plots(self, save_format: str = 'pdf') -> Dict[str, Optional[str]]:
        """
        Generate all experimental plots for IBM quantum data.
        
        Args:
            save_format: File format for saving figures
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        print("\n" + "="*70)
        print("GENERATING IBM EXPERIMENTAL FIGURES")
        print("="*70)
        
        plots = {}
        
        if self.data.empty:
            print("No experimental data available")
            return plots
        
        # Check available noise types
        available_noise_types = self.data['noise_type'].unique()
        
        # 1. Combined threshold plot (both noise types on same plot)
        print("\n1. Combined threshold plot (both noise types)...")
        plots['threshold_combined_exp'] = self.plot_threshold_vs_M_combined_experimental(save_format)
        
        # 2. Fidelity vs M plots (separate for each noise type)
        print("\n2. Fidelity vs M plots...")
        if 'depolarizing' in available_noise_types:
            plots['fidelity_vs_M_depol_exp'] = self.plot_fidelity_vs_M_experimental('depolarizing', save_format)
        if 'dephasing' in available_noise_types:
            plots['fidelity_vs_M_dephase_exp'] = self.plot_fidelity_vs_M_experimental('dephasing', save_format)
        
        # 3. Combined fidelity plot (2x2 grid)
        print("\n3. Combined fidelity plot (2x2 grid)...")
        plots['fidelity_combined_M_exp'] = self.plot_fidelity_combined_M_experimental(save_format)
        
        # Summary
        successful = [name for name, path in plots.items() if path is not None]
        print(f"\n{len(successful)}/{len(plots)} experimental plots generated successfully")
        print(f"Figures saved to: {self.figures_dir}")
        
        return plots

    def print_data_summary(self):
        """Print a summary of the loaded experimental data."""
        if self.data.empty:
            print("No data loaded")
            return
        
        print("\n" + "="*50)
        print("IBM EXPERIMENTAL DATA SUMMARY")
        print("="*50)
        print(f"Total runs: {len(self.data)}")
        print(f"Date range: {self.data['run_id'].iloc[0]} to {self.data['run_id'].iloc[-1]}")
        
        print(f"\nNoise types: {list(self.data['noise_type'].unique())}")
        print(f"M values: {sorted(self.data['M'].unique())}")
        print(f"N values: {sorted(self.data['N'].unique())}")
        print(f"p values: {sorted(self.data['p'].unique())}")
        
        if 'backend_name' in self.data.columns:
            print(f"Backend: {self.data['backend_name'].iloc[0]}")
        
        print(f"\nFidelity range: {self.data['final_fidelity'].min():.3f} - {self.data['final_fidelity'].max():.3f}")
        
        # Data completeness
        print(f"\nData completeness:")
        for noise in self.data['noise_type'].unique():
            for M in sorted(self.data['M'].unique()):
                count = len(self.data[(self.data['noise_type'] == noise) & (self.data['M'] == M)])
                print(f"  {noise}, M={M}: {count} runs")


def main():
    """Main function to generate all experimental plots."""
    import sys
    
    # Default parameters
    csv_path = "data/IBMQ/IBM_results_all_ibm_torino.csv"
    # csv_path = "data/IBMQ/data_OLD.csv"
    # csv_path = "data/IBMQ/Aer_results_all_20251202_171217.csv"
    figures_dir = "figures/experimental_ibm"
    save_format = "pdf"
    
    # Command line arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        figures_dir = sys.argv[2]
    if len(sys.argv) > 3:
        save_format = sys.argv[3]
    
    # Create plotter and generate plots
    plotter = IBMExperimentalPlotter(csv_path, figures_dir)
    
    # Show data summary
    plotter.print_data_summary()
    
    # Generate all plots
    plots = plotter.generate_all_experimental_plots(save_format)
    
    print("\n" + "="*70)
    print("EXPERIMENTAL PLOTTING COMPLETE")
    print("="*70)
    
    return plots


if __name__ == "__main__":
    main()