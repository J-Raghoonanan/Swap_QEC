#!/bin/bash
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR="/data/rho2_sim"
FIG_DIR="/figures/rho2_results"

mkdir -p "$DATA_DIR"
mkdir -p "$FIG_DIR"

# ── Dataset 1: Depolarizing noise ──────────────────────────────────────────
echo "=== [1/5] Depolarizing noise ==="
python -m src.simulation.rho2_sims.main_grid_run \
    --out "$DATA_DIR" \
    --noise depol \
    --m-values 1 2 3 4 5 \
    --iterative

# ── Dataset 2: Z-dephasing with Clifford twirling ─────────────────────────
echo "=== [2/5] Z-dephasing with twirl ==="
python -m src.simulation.rho2_sims.main_grid_run \
    --out "$DATA_DIR" \
    --noise z \
    --m-values 1 2 3 4 5 \
    --iterative

# ── Dataset 3: Z-dephasing, no twirl ──────────────────────────────────────
echo "=== [3/5] Z-dephasing no twirl ==="
python -m src.simulation.rho2_sims.main_grid_run \
    --out "$DATA_DIR" \
    --noise z \
    --m-values 1 2 3 4 5 \
    --iterative \
    --no-twirl

# ── Dataset 4: Single-qubit product state, Z-dephasing, no twirl ──────────
echo "=== [4/5] Single-qubit product state ==="
python -m src.simulation.rho2_sims.main_grid_run \
    --out "$DATA_DIR" \
    --noise z \
    --m-values 1 2 3 4 5 \
    --iterative \
    --no-twirl \
    --target single_qubit_product

# ── Dataset 5: Approximate twirl (subset_fraction=0.2) ────────────────────
echo "=== [5/5] Approximate twirl ==="
python -m src.simulation.rho2_approx_twirl_sim.main_grid_run \
    --out "$DATA_DIR" \
    --noise z \
    --m-values 1 5 \
    --iterative \
    --subset-fraction 0.2 \
    --subset-seed 42

# ── Figures ────────────────────────────────────────────────────────────────
echo "=== Generating figures ==="
python figures/rho2_plots.py \
    --data-dir "$DATA_DIR" \
    --figures-dir  "$FIG_DIR"

echo "=== Done. Results written to $DATA_DIR and $FIG_DIR ==="