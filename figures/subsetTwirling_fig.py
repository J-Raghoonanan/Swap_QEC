"""
Figure generation for subset twirling purification simulation data.

Loads CSV data from subset twirling simulations and creates publication-quality figures
showing fidelity vs iterations and gamma vs p in a combined 2x2 grid layout.

This is adapted from the original iterative purification plotter to work with
subset twirling data that may have different file naming conventions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Literal
from pathlib import Path

from dataclasses import dataclass

try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

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


class SubsetTwirlingPlotter:
    """Generate figures from subset twirling purification CSV data."""
    
    def __init__(self, data_dir: str = "data/subsetTwirling_simulations", 
                 figures_dir: str = "figures/subsetTwirl",
                 subset_fraction: float = 0.5):
        """
        Initialize plotter for subset twirling data.
        
        Parameters
        ----------
        data_dir : str
            Directory containing CSV files
        figures_dir : str
            Directory to save generated figures
        subset_fraction : float
            Subset fraction to look for in filenames (e.g., 0.5 for 50%)
        """
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.subset_fraction = subset_fraction
        
        # Load data - try different naming patterns
        self.depol_steps = self._load_data('depolarizing', 'steps')
        self.depol_finals = self._load_data('depolarizing', 'finals')
        self.dephase_steps = self._load_data('dephase_z', 'steps')
        self.dephase_finals = self._load_data('dephase_z', 'finals')
        
        self.subsetTwirl_dephase_steps_debug = self._load_csv_from(self.data_dir, f"steps_circuit_dephase_z__subset{self.subset_fraction:.2f}_DEBUG.csv")
        self.subsetTwirl_dephase_finals_debug = self._load_csv_from(self.data_dir, f"finals_circuit_dephase_z__subset{self.subset_fraction:.2f}_DEBUG.csv")
        
        print(f"Loaded subset twirling data (fraction={subset_fraction}):")
        print(f"  Depolarizing steps: {len(self.depol_steps)} rows")
        print(f"  Depolarizing finals: {len(self.depol_finals)} rows")
        print(f"  Dephasing steps: {len(self.dephase_steps)} rows")
        print(f"  Dephasing finals: {len(self.dephase_finals)} rows")
        
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
    
    def _load_data(self, noise_type: str, data_type: str) -> pd.DataFrame:
        """
        Load CSV file with flexible naming pattern matching.
        
        Tries multiple filename patterns:
        1. With subset indicator: steps_circuit_dephase_z__subset0.50.csv
        2. Without subset indicator: steps_circuit_dephase_z.csv
        3. Simple pattern: steps_circuit_depolarizing.csv
        """
        patterns = []
        
        # Pattern 1: With subset fraction
        if self.subset_fraction < 1.0:
            subset_str = f"subset{self.subset_fraction:.2f}"
            patterns.append(f"{data_type}_circuit_{noise_type}__{subset_str}.csv")
        
        # Pattern 2: Without subset indicator (for fraction=1.0 or files without suffix)
        patterns.append(f"{data_type}_circuit_{noise_type}.csv")
        
        # Pattern 3: Simple pattern
        patterns.append(f"{data_type}_circuit_{noise_type}.csv")
        
        # Try each pattern
        for pattern in patterns:
            filepath = self.data_dir / pattern
            if filepath.exists():
                print(f"  Loading {filepath.name}...")
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
        
        print(f"  Warning: No file found for {noise_type} {data_type}")
        print(f"    Tried patterns: {patterns}")
        return pd.DataFrame()
    
    def plot_combined_fidelity_and_gamma_2x2_grid(self, noise_type: str, save_format: str = 'pdf', 
                                              *, p_center: float = 0.60, p_halfwidth: float = 0.05):
        """
        Combined 2x2 grid plot identical to original but for subset twirling data:
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
        else:  # dephasing
            df_steps = self.dephase_steps
            df_steps_debug = self.subsetTwirl_dephase_steps_debug
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
        # fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
        df_fid_filtered_debug = df_steps_debug[(df_steps_debug['M'] == 5) & (df_steps_debug[twirling_col] == twirling_filter)].copy()
        print(f"Fidelity plots: {len(df_fid_filtered)} rows after M=5, twirling filter")
        
        if not df_fid_filtered.empty:
            l_values = sorted(df_fid_filtered['purification_level'].unique())
            p_subset_fid = [0.1, 0.3]  # For top row
            
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
                    
                    # Plot
                    if l_val == 0:
                        ax.plot(x, y, linestyle='dotted', marker=_mk(j), 
                            color=colors[j % len(colors)], linewidth=2, markersize=12, 
                            alpha=0.8, label=rf'No QEC')
                    elif l_val <= 3:
                        ax.plot(x, y, linestyle='-', marker=_mk(j), 
                            color=colors[j % len(colors)], linewidth=2, markersize=12, 
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
                if i == 1:
                    ax.legend(fontsize=28, loc='best', frameon=False)
                
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
        
        df_tw_debug = df_steps_debug[df_steps_debug[twirling_col] == twirling_filter].copy()
        df_t1_debug = df_tw_debug[df_tw_debug['iteration'] == 1].copy()
        
        if not df_t1.empty:
            # Calculate gamma = 1 - F(t=1)
            grp_cols = ['M', 'purification_level', 'p']
            df_gamma = (df_t1.groupby(grp_cols, as_index=False)['fidelity']
                        .mean().rename(columns={'fidelity': 'F_t1'}))
            df_gamma['gamma'] = 1.0 - df_gamma['F_t1']
            
            df_gamma_debug = (df_t1_debug.groupby(grp_cols, as_index=False)['fidelity']
                        .mean().rename(columns={'fidelity': 'F_t1'}))
            df_gamma_debug['gamma'] = 1.0 - df_gamma_debug['F_t1']
            
            l_values_gamma = sorted(df_gamma['purification_level'].unique())
            l_to_color_idx = {l_val: idx for idx, l_val in enumerate(l_values_gamma)}
            
            # Exclusion set for main plots
            exclude = {0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79}
            
            # Inset zoom parameters
            p_min_zoom = max(0.0, p_center - p_halfwidth)
            p_max_zoom = min(1.0, p_center + p_halfwidth)
            # inset_ylim = (0.95, 0.98)  # For M=5 inset
            inset_ylim = (0.90, 0.96)  # For M=5 inset
            
            target_Ms = [1, 5]
            
            for j, M in enumerate(target_Ms):
                ax = axes[1, j]  # Bottom row
                
                df_M = df_gamma[df_gamma['M'] == M].copy()
                df_M_debug = df_gamma_debug[df_gamma_debug['M'] == M].copy()
                
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
                    df_ml_debug = df_M_debug[df_M_debug['purification_level'] == l_val].copy()
                    
                    if df_ml.empty:
                        continue
                    
                    # Apply exclusion filter
                    p_round = df_ml['p'].round(2)
                    df_main = df_ml[~p_round.isin(exclude)].copy()
                    df_main_debug = df_ml_debug[~p_round.isin(exclude)].copy()
                    
                    if df_main.empty:
                        continue
                    
                    # Sort and plot
                    df_main = df_main.sort_values('p')
                    df_main_debug = df_main_debug.sort_values('p')
                    linestyle = ':' if int(l_val) == 0 else '-'
                    cidx = l_to_color_idx[l_val]
                    
                    if l_val == 0:
                        label = rf'No QEC'
                    else:
                        label = rf'$\ell={l_val}$'
                    
                    if l_val <= 3: # Only plot up to ℓ=3 in main plot
                        if M==1:
                            line, = ax.plot(df_main_debug['p'], df_main_debug['gamma'],
                                    linestyle=linestyle, marker=_mk(cidx),
                                    linewidth=2.5, markersize=12, alpha=0.85,
                                    color=colors[cidx % len(colors)],
                                    label=f"{label} (debug)")
                        else:
                            line, = ax.plot(df_main['p'], df_main['gamma'],
                                        linestyle=linestyle, marker=_mk(cidx),
                                        linewidth=2.5, markersize=12, alpha=0.85,
                                        color=colors[cidx % len(colors)],
                                        label=label)
                        handles.append(line)
                        labels.append(label)
                
                # Legend only on M=1 (left plot)
                if M == 1:
                    ax.legend(handles, labels, fontsize=28, loc='best', frameon=False)
                
                # Subplot label
                ax.text(0.08, 0.98, subplot_labels[1][j], transform=ax.transAxes,
                    fontsize=50, fontweight='bold', va='top', ha='right')
                
                # ========================
                # INSET ONLY FOR M=5
                # ========================
                if M == 5:
                    axins = inset_axes(ax, width='48%', height='42%', 
                                    loc='lower right', borderpad=1.1)
                    
                    # Plot full curves in inset
                    for l_val in l_values_gamma:
                        df_ml = df_M[df_M['purification_level'] == l_val].copy()
                        if df_ml.empty:
                            continue
                        
                        df_ml = df_ml.sort_values('p')
                        linestyle = ':' if int(l_val) == 0 else '-'
                        cidx = l_to_color_idx[l_val]
                        
                        x = df_ml['p'].to_numpy(dtype=float)
                        y = df_ml['gamma'].to_numpy(dtype=float)
                        
                        # Filter positive values for log scale
                        mask = y > 0.0
                        if mask.sum() < 2:
                            continue
                        
                        if l_val <= 3:
                            axins.plot(x[mask], y[mask], linestyle=linestyle, marker=_mk(cidx),
                                    linewidth=2.0, markersize=6, alpha=0.9,
                                    color=colors[cidx % len(colors)])
                    
                    # Inset formatting
                    axins.set_xlim(p_min_zoom, p_max_zoom)
                    axins.set_yscale('log')
                    
                    # Set y-limits for crossover visibility
                    ymin, ymax = inset_ylim
                    ymin = max(float(ymin), 1e-12)
                    ymax = max(float(ymax), ymin * 1.01)
                    axins.set_ylim(ymin, ymax)
                    
                    # Inset tick formatting
                    axins.tick_params(axis='both', which='major', labelsize=15)
                    axins.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                    axins.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
                    axins.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
                    axins.yaxis.set_minor_formatter(mticker.NullFormatter())
        
        # Overall formatting
        plt.tight_layout()
        
        # Save
        subset_str = f"_subset{self.subset_fraction:.2f}" if self.subset_fraction < 1.0 else ""
        filename = f"combined_fidelity_gamma_2x2_{noise_type}{subset_str}.{save_format}"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        return str(filepath)


def main():
    """Main function to generate subset twirling plots."""
    import sys
    
    # Default parameters
    data_dir = "data/subsetTwirling_simulations"
    figures_dir = "figures/subsetTwirl"
    subset_fraction = 0.3
    save_format = "pdf"
    
    # Parse command line arguments
    if '--data-dir' in sys.argv:
        idx = sys.argv.index('--data-dir')
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    
    if '--figures-dir' in sys.argv:
        idx = sys.argv.index('--figures-dir')
        if idx + 1 < len(sys.argv):
            figures_dir = sys.argv[idx + 1]
    
    if '--subset-fraction' in sys.argv:
        idx = sys.argv.index('--subset-fraction')
        if idx + 1 < len(sys.argv):
            subset_fraction = float(sys.argv[idx + 1])
    
    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        if idx + 1 < len(sys.argv):
            save_format = sys.argv[idx + 1]
    
    print("\n" + "="*70)
    print("GENERATING SUBSET TWIRLING FIGURES")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Figures directory: {figures_dir}")
    print(f"Subset fraction: {subset_fraction}")
    print(f"Save format: {save_format}")
    print("="*70 + "\n")
    
    # Create plotter
    plotter = SubsetTwirlingPlotter(data_dir, figures_dir, subset_fraction)
    
    # Generate plots for both noise types
    plots = {}
    
    # print("\n1. Combined fidelity and gamma plot (depolarizing)...")
    # plots['combined_depol'] = plotter.plot_combined_fidelity_and_gamma_2x2_grid(
    #     'depolarizing', save_format)
    
    print("\n2. Combined fidelity and gamma plot (dephasing)...")
    plots['combined_dephase'] = plotter.plot_combined_fidelity_and_gamma_2x2_grid(
        'dephasing', save_format)
    
    # Summary
    successful = [name for name, path in plots.items() if path is not None]
    print(f"\n{len(successful)}/{len(plots)} plots generated successfully")
    print(f"Figures saved to: {figures_dir}")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return plots


if __name__ == "__main__":
    main()