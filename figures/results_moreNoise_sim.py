"""
Figure generation for iterative purification simulation data.
Loads CSV data from iterative purification simulations and creates publication-quality figures
showing final fidelity vs rounds of iteration for different purification levels (ℓ).
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


class IterativePurificationPlotter:
    """Generate figures from iterative purification CSV data."""
    
    def __init__(self, data_dir: str = "data/simulations_moreNoise", 
                 figures_dir: str = "figures/iterative_results"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.depol_finals = self._load_csv('finals_circuit_depolarizing.csv')
        self.depol_steps = self._load_csv('steps_circuit_depolarizing.csv')
        self.dephase_finals = self._load_csv('finals_circuit_dephase_z.csv')
        self.dephase_steps = self._load_csv('steps_circuit_dephase_z.csv')
        
        print(f"Loaded iterative purification data:")
        print(f"  Depolarizing finals: {len(self.depol_finals)} runs")
        print(f"  Depolarizing steps: {len(self.depol_steps)} iteration steps")
        print(f"  Dephasing finals: {len(self.dephase_finals)} runs")
        print(f"  Dephasing steps: {len(self.dephase_steps)} iteration steps")
    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file if it exists."""
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
        
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
    
    def plot_fidelity_vs_iterations(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot final fidelity vs rounds of iteration for different ℓ values (M=1).
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            color = COLORS['depolarizing']
            twirling_filter = False
        else:  # dephasing
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iterative columns for {noise_type}")
            print(f"Available columns: {list(df.columns)}")
            return None
    
        # Filter M=1 AND twirling condition
        df_filtered = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_filtered.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique purification levels (ℓ values)
        l_values = sorted(df_filtered['purification_level'].unique())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']
    
        # Choose a representative p value (middle of available range)
        p_values = sorted(df_filtered['p'].unique()) if 'p' in df_filtered.columns else []
        if p_values:
            mid_p = p_values[len(p_values) // 2]
            df_p = df_filtered[df_filtered['p'] == mid_p].copy()
        else:
            df_p = df_filtered.copy()
    
        for i, l_val in enumerate(l_values):
            df_l = df_p[df_p['purification_level'] == l_val].copy()
        
            if len(df_l) > 0:
                # Sort by iteration number
                df_l = df_l.sort_values('iteration')
                
                ax.plot(df_l['iteration'], df_l['fidelity'],
                        linestyle='-', marker=_mk(i),
                        color=colors[i % len(colors)], linewidth=3, markersize=8,
                        label=rf'$\ell$ = {l_val}', alpha=0.8)
    
        ax.set_xlabel('Iteration Round', fontsize=30)
        ax.set_ylabel('Fidelity', fontsize=30)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        if p_values:
            ax.set_title(f'Iterative Purification\n({title_str} Noise, M=1, p={mid_p:.2f})', fontsize=40)
        else:
            ax.set_title(f'Iterative Purification\n({title_str} Noise, M=1)', fontsize=40)
    
        ax.legend(fontsize=14, loc='best')
        ax.set_ylim(0.0, 1.05)
        
        if not df_filtered.empty:
            max_iter = df_filtered['iteration'].max()
            ax.set_xlim(0.5, max_iter + 0.5)
    
        plt.tight_layout()
    
        filename = f"fidelity_vs_iterations_{noise_type}_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)


    def plot_fidelity_vs_iterations_multiple_p(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot fidelity vs iterations for different ℓ values, with 2x2 subplots for different p values.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            color = COLORS['depolarizing']
            twirling_filter = False
        else:  # dephasing
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iterative columns for {noise_type}")
            return None
    
        # Filter M=1 AND twirling condition
        df_filtered = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_filtered.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        # Get unique values
        l_values = sorted(df_filtered['purification_level'].unique())
        p_values = sorted(df_filtered['p'].unique()) if 'p' in df_filtered.columns else []
        
        if len(p_values) == 0:
            print(f"No p values found for {noise_type}")
            return None
        
        # Select up to 4 p values for 2x2 grid
        # if len(p_values) > 4:
        #     # Take evenly spaced p values
        #     indices = np.linspace(0, len(p_values)-1, 4).astype(int)
        #     p_subset = [p_values[i] for i in indices]
        # else:
        #     p_subset = p_values
        p_subset = [0.1, 0.3, 0.5, 0.7]
    
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']
    
        # Plot each p value (up to 4)
        for i, p_val in enumerate(p_subset[:4]):
            ax = axes[i]
            df_p = df_filtered[df_filtered['p'] == p_val].copy()
            
            if df_p.empty:
                ax.text(0.5, 0.5, f'No data\nfor p = {p_val:.2f}',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, alpha=0.7)
                continue
            
            # Plot curves for different ℓ values
            for j, l_val in enumerate(l_values):
                df_l = df_p[df_p['purification_level'] == l_val].copy()
                
                if df_l.empty:
                    continue
                
                # Sort by iteration
                df_l = df_l.sort_values('iteration')
                
                # Plot trajectory
                ax.plot(df_l['iteration'], df_l['fidelity'],
                       linestyle='-', marker=_mk(j), color=colors[j % len(colors)],
                       linewidth=2, markersize=8, alpha=0.8,
                       label=rf'$\ell$ = {l_val}')
            
            # Formatting for this subplot
            ax.set_title(f'p = {p_val:.2f}', fontsize=24)
            ax.set_ylim(0.0, 1.05)
            
            if not df_p.empty:
                max_iter = df_p['iteration'].max()
                ax.set_xlim(0.5, max_iter + 0.5)
            
            # Labels
            if i >= 2:  # Bottom row
                ax.set_xlabel('Iteration Round', fontsize=20)
            if i % 2 == 0:  # Left column
                ax.set_ylabel('Fidelity', fontsize=20)
            
            # Add legend to first subplot
            if i == 0 and len(l_values) > 0:
                ax.legend(fontsize=12, loc='best')
        
        # Hide unused subplots
        for i in range(len(p_subset), 4):
            axes[i].axis('off')
        
        # Main title
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        fig.suptitle(f'Iterative Purification - {title_str} Noise (M=1)', fontsize=32, y=0.96)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        filename = f"fidelity_vs_iterations_{noise_type}_multi_p_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)


    def plot_fidelity_vs_iterations_selected_p(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot fidelity vs iterations with multiple p value curves on the same plot.
        Uses p = 0.1, 0.5, 0.7 and shows different ℓ values as separate plots.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps
            color = COLORS['depolarizing']
            twirling_filter = False
        else:  # dephasing
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iterative columns for {noise_type}")
            return None
    
        # Filter M=1 AND twirling condition
        df_filtered = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_filtered.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        # Selected p values to plot
        target_p_values = [0.1, 0.5, 0.7]
        available_p_values = sorted(df_filtered['p'].unique()) if 'p' in df_filtered.columns else []
        
        # Find closest available p values to targets
        selected_p_values = []
        for target_p in target_p_values:
            if available_p_values:
                closest_p = min(available_p_values, key=lambda x: abs(x - target_p))
                if abs(closest_p - target_p) < 0.05:  # Within 5% tolerance
                    selected_p_values.append(closest_p)
        
        if len(selected_p_values) == 0:
            print(f"No p values close to targets {target_p_values} for {noise_type}")
            return None
        
        # Get unique ℓ values
        l_values = sorted(df_filtered['purification_level'].unique())
        
        if len(l_values) == 0:
            print(f"No purification levels found for {noise_type}")
            return None
        
        # Create subplots for each ℓ value
        if len(l_values) <= 2:
            fig, axes = plt.subplots(1, len(l_values), figsize=(6*len(l_values), 8))
            if len(l_values) == 1:
                axes = [axes]
        else:
            rows = int(np.ceil(len(l_values) / 2))
            fig, axes = plt.subplots(rows, 2, figsize=(12, 6*rows))
            axes = axes.flatten()
        
        # Colors for different p values
        p_colors = ['#2E86AB', '#F18F01', '#A23B72']  # Blue, orange, purple
        
        # Plot each ℓ value
        for i, l_val in enumerate(l_values):
            ax = axes[i]
            df_l = df_filtered[df_filtered['purification_level'] == l_val].copy()
            
            if df_l.empty:
                ax.text(0.5, 0.5, f'No data\nfor ℓ = {l_val}',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, alpha=0.7)
                continue
            
            # Plot trajectories for different p values
            for j, p_val in enumerate(selected_p_values):
                df_p = df_l[df_l['p'] == p_val].copy()
                
                if df_p.empty:
                    continue
                
                # Sort by iteration
                df_p = df_p.sort_values('iteration')
                
                # Plot trajectory
                ax.plot(df_p['iteration'], df_p['fidelity'],
                       linestyle='-', marker=_mk(j), color=p_colors[j % len(p_colors)],
                       linewidth=3, markersize=10, alpha=0.8,
                       label=f'p = {p_val:.1f}')
            
            # Formatting for this subplot
            ax.set_title(rf'$\ell$ = {l_val}', fontsize=30)
            ax.set_ylim(0.0, 1.05)
            
            if not df_l.empty:
                max_iter = df_l['iteration'].max()
                ax.set_xlim(0.5, max_iter + 0.5)
            
            # Labels
            if i >= len(l_values) - 2 or (len(l_values) > 2 and i >= len(l_values) - 2):  # Bottom row
                ax.set_xlabel('Iteration Round', fontsize=24)
            if i % 2 == 0 or len(l_values) == 1:  # Left column or single plot
                ax.set_ylabel('Fidelity', fontsize=24)
            
            # Add legend to first subplot
            if i == 0 and len(selected_p_values) > 0:
                ax.legend(fontsize=14, loc='best')
        
        # Hide unused subplots
        for i in range(len(l_values), len(axes)):
            axes[i].axis('off')
        
        # Main title
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        fig.suptitle(f'Iterative Purification - {title_str} Noise (M=1)\nMultiple Error Rates', fontsize=32, y=0.96)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        filename = f"fidelity_vs_iterations_{noise_type}_selected_p_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)


    def plot_final_fidelity_vs_p(self, noise_type: str, save_format: str = 'pdf') -> Optional[str]:
        """
        Plot fidelity vs physical error rate p for different ℓ values.
        Shows two curves per ℓ: 1st iteration and last iteration.
        """
        # Select data and set twirling filter
        if noise_type == 'depolarizing':
            df = self.depol_steps  # Use steps data to get iteration-level info
            color = COLORS['depolarizing']
            twirling_filter = False
            param_label = r'$p$'
        else:  # dephasing
            df = self.dephase_steps
            color = COLORS['dephasing']
            twirling_filter = True
            param_label = r'$p$'
    
        if df.empty:
            print(f"No steps data for {noise_type}")
            return None
    
        # Check for iterative columns
        if 'iteration' not in df.columns or 'purification_level' not in df.columns:
            print(f"Missing iteration/purification_level columns for {noise_type}")
            return None
    
        # Filter M=1 AND twirling condition
        df_filtered = df[(df['M'] == 1) & (df['twirling_enabled'] == twirling_filter)].copy()
    
        if df_filtered.empty:
            print(f"No M=1 data for {noise_type} with twirling={twirling_filter}")
            return None
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Get unique purification levels
        l_values = sorted(df_filtered['purification_level'].unique())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'saddlebrown', 'deeppink', 'darkslategrey', 'fuchsia', 'gold']
    
        for i, l_val in enumerate(l_values):
            df_l = df_filtered[df_filtered['purification_level'] == l_val].copy()
        
            if len(df_l) > 0:
                # For each p value, get the 1st and last iteration
                p_values = sorted(df_l['p'].unique())
                
                first_iter_data = []
                last_iter_data = []
                
                for p_val in p_values:
                    df_p = df_l[df_l['p'] == p_val].copy()
                    
                    if df_p.empty:
                        continue
                    
                    # Sort by iteration to get first and last
                    df_p = df_p.sort_values('iteration')
                    
                    # Get first iteration (iteration = 1)
                    first_iter_row = df_p[df_p['iteration'] == 1]
                    if not first_iter_row.empty:
                        first_iter_data.append({
                            'p': p_val,
                            'fidelity': first_iter_row['fidelity'].iloc[0]
                        })
                    
                    # Get last iteration
                    last_iter_row = df_p.iloc[-1]  # Last row after sorting by iteration
                    last_iter_data.append({
                        'p': p_val,
                        'fidelity': last_iter_row['fidelity']
                    })
                
                # Convert to DataFrames for easier plotting
                if first_iter_data:
                    df_first = pd.DataFrame(first_iter_data).sort_values('p')
                    ax.plot(df_first['p'], df_first['fidelity'],
                            linestyle='--', marker=_mk(i), 
                            color=colors[i % len(colors)], linewidth=2, markersize=6,
                            label=rf'$\ell$ = {l_val} (1st iter)', alpha=0.7)
                
                if last_iter_data:
                    df_last = pd.DataFrame(last_iter_data).sort_values('p')
                    ax.plot(df_last['p'], df_last['fidelity'],
                            linestyle='-', marker=_mk(i),
                            color=colors[i % len(colors)], linewidth=3, markersize=8,
                            label=rf'$\ell$ = {l_val} (final iter)', alpha=0.8)
    
        ax.set_xlabel(f'Physical Error Rate, {param_label}', fontsize=30)
        ax.set_ylabel('Fidelity', fontsize=30)
    
        title_str = 'Depolarizing' if noise_type == 'depolarizing' else 'Dephasing'
        ax.set_title(f'Iterative PEC Performance\n({title_str} Noise, M=1)', fontsize=40)
    
        ax.legend(fontsize=12, loc='best')
        ax.set_xlim(0.05, 1.0)
        ax.set_ylim(0.0, 1.05)
    
        plt.tight_layout()
    
        filename = f"final_fidelity_vs_p_{noise_type}_M1.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Saved {filename}")
        return str(filepath)

    
    def generate_all_plots(self, save_format: str = 'pdf') -> Dict[str, Optional[str]]:
        """Generate all iterative purification plots."""
        print("\n" + "="*70)
        print("GENERATING ITERATIVE PURIFICATION FIGURES")
        print("="*70)
        
        plots = {}
        
        # Fidelity vs iterations (single p value)
        print("\n1. Fidelity vs iterations (representative p)...")
        plots['fidelity_vs_iter_depol'] = self.plot_fidelity_vs_iterations('depolarizing', save_format)
        plots['fidelity_vs_iter_dephase'] = self.plot_fidelity_vs_iterations('dephasing', save_format)
        
        # Fidelity vs iterations (2x2 grid of p values)
        print("\n2. Fidelity vs iterations (2x2 grid of p values)...")
        plots['fidelity_vs_iter_multi_depol'] = self.plot_fidelity_vs_iterations_multiple_p('depolarizing', save_format)
        plots['fidelity_vs_iter_multi_dephase'] = self.plot_fidelity_vs_iterations_multiple_p('dephasing', save_format)
        
        # Fidelity vs iterations (selected p values: 0.1, 0.5, 0.7)
        print("\n3. Fidelity vs iterations (selected p values: 0.1, 0.5, 0.7)...")
        plots['fidelity_vs_iter_selected_depol'] = self.plot_fidelity_vs_iterations_selected_p('depolarizing', save_format)
        plots['fidelity_vs_iter_selected_dephase'] = self.plot_fidelity_vs_iterations_selected_p('dephasing', save_format)
        
        # Final fidelity vs p (threshold-like plots)
        print("\n4. Final fidelity vs physical error rate...")
        plots['final_fidelity_vs_p_depol'] = self.plot_final_fidelity_vs_p('depolarizing', save_format)
        plots['final_fidelity_vs_p_dephase'] = self.plot_final_fidelity_vs_p('dephasing', save_format)
        
        # Summary
        successful = [name for name, path in plots.items() if path is not None]
        print(f"\n{len(successful)}/{len(plots)} plots generated successfully")
        print(f"Figures saved to: {self.figures_dir}")
        
        return plots


def main():
    """Main function."""
    import sys
    
    data_dir = "data/simulations_moreNoise"
    figures_dir = "figures/iterative_results"
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
    
    plotter = IterativePurificationPlotter(data_dir, figures_dir)
    plots = plotter.generate_all_plots(save_format)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return plots


if __name__ == "__main__":
    main()