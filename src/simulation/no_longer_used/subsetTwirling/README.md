# Subset Twirling for SWAP-based Purification

This directory contains a resource-efficient implementation of Clifford twirling for the SWAP-based quantum error correction scheme. It allows using only a **subset** of the full 3^M Clifford gate combinations, reducing computational cost while maintaining effective noise mitigation.

## Key Features

### Subset Twirling Parameters

The `TwirlingSpec` configuration now includes:

```python
@dataclass
class TwirlingSpec:
    enabled: bool = True
    mode: Literal["random", "cyclic"] = "cyclic"
    seed: Optional[int] = None
    
    # NEW: Subset parameters
    subset_fraction: float = 1.0        # Fraction of 3^M gates to use (0.0 to 1.0)
    subset_mode: Literal["random", "first_k"] = "random"  # Selection strategy
    subset_seed: Optional[int] = None   # Seed for subset selection
```

### How It Works

For full Clifford twirling of M-qubit dephasing noise, the ideal approach averages over all 3^M combinations:

```
E_twirled(ρ) = (1/3^M) Σ_{C ∈ {I,H,HS}^⊗M} C† E(C ρ C†) C
```

This is expensive: M=5 requires 243 combinations, M=6 requires 729.

**Subset twirling** uses only a fraction of these combinations:

```
E_subset(ρ) = (1/K) Σ_{C ∈ subset} C† E(C ρ C†) C
```

where K = ⌈fraction × 3^M⌉.

### Subset Selection Modes

1. **Random sampling** (`subset_mode="random"`):
   - Uniformly samples K combinations without replacement
   - Provides unbiased estimate of full average
   - Use `subset_seed` for reproducibility

2. **Deterministic first-k** (`subset_mode="first_k"`):
   - Takes first K combinations in lexicographic order
   - Deterministic and reproducible
   - May introduce bias for very small fractions

## Computational Savings

| M  | Full (3^M) | 50% subset | 25% subset | Savings (50%) |
|----|------------|------------|------------|---------------|
| 3  | 27         | 14         | 7          | 1.9x          |
| 4  | 81         | 41         | 21         | 2.0x          |
| 5  | 243        | 122        | 61         | 2.0x          |
| 6  | 729        | 365        | 183        | 2.0x          |

## Usage Examples

### Command Line

```bash
# Use 50% of Clifford combinations (random sampling)
python -m src.simulation.subsetTwirling.main_grid_run \
    --out data/subsetTwirling_simulations \
    --noise z \
    --m-values 5 \
    --subset-fraction 0.5 \
    --subset-mode random

# Use 25% (deterministic selection)
python -m src.simulation.subsetTwirling.main_grid_run \
    --out data/subsetTwirling_simulations \
    --noise z \
    --m-values 5 \
    --subset-fraction 0.25 \
    --subset-mode first_k

# Full twirling (100%)
python -m src.simulation.subsetTwirling.main_grid_run \
    --out data/subsetTwirling_simulations \
    --noise z \
    --m-values 5 \
    --subset-fraction 1.0
```

### Programmatic

```python
from src.simulation.subsetTwirling.configs import RunSpec, TwirlingSpec, TargetSpec, NoiseSpec, AASpec
from src.simulation.subsetTwirling.configs import NoiseType, NoiseMode, StateKind

# Configure subset twirling
twirling = TwirlingSpec(
    enabled=True,
    subset_fraction=0.5,      # Use 50% of combinations
    subset_mode="random",
    subset_seed=42,           # For reproducibility
)

spec = RunSpec(
    target=TargetSpec(M=5, kind=StateKind.hadamard),
    noise=NoiseSpec(noise_type=NoiseType.dephase_z, p=0.3),
    aa=AASpec(),
    twirling=twirling,
    N=64,
    iterative_noise=True,
    purification_level=3,
)

from src.simulation.subsetTwirling.streaming_runner import run_and_save
run_and_save(spec)
```

## Implementation Details

### Modified Files

1. **configs.py**: Added subset parameters to `TwirlingSpec`
2. **noise_engine.py**: 
   - New function `_get_clifford_subset()` generates subset
   - New function `_apply_noise_with_twirling()` averages over subset
   - Updated `apply_noise_to_density_matrix()` to use subset twirling
3. **main_grid_run.py**: Added CLI arguments for subset control
4. **streaming_runner.py**: Updated CSV filenames to indicate subset usage

### Unchanged Files

- **state_factory.py**: Pure state preparation (independent of noise)
- **amplified_swap.py**: SWAP test implementation (independent of noise)

## Validation

To validate that subset twirling provides similar results to full twirling:

```bash
# Run with full twirling
python -m src.simulation.subsetTwirling.main_grid_run \
    --noise z --m-values 3 --subset-fraction 1.0 \
    --out data/validation_full

# Run with 50% subset
python -m src.simulation.subsetTwirling.main_grid_run \
    --noise z --m-values 3 --subset-fraction 0.5 \
    --out data/validation_subset

# Compare results in your analysis scripts
```

Expected behavior:
- **High fractions (≥0.5)**: Should closely match full twirling
- **Low fractions (<0.3)**: May show increased variance but still effective
- **Very low fractions (<0.1)**: May deviate significantly from full twirling

## When to Use Subset Twirling

**Use full twirling (fraction=1.0) when:**
- M ≤ 4 (computational cost is reasonable)
- You need exact convergence guarantees
- You're validating theoretical predictions

**Use subset twirling (fraction=0.3-0.7) when:**
- M ≥ 5 (full twirling becomes expensive)
- You're exploring parameter space with many runs
- Approximate noise mitigation is sufficient
- Resource constraints are tight

**Avoid very small fractions (<0.2) when:**
- Precise quantitative results are needed
- Comparing against theoretical bounds

## Directory Structure

```
src/simulation/subsetTwirling/
├── __init__.py              # Module initialization
├── README.md                # This file
├── configs.py               # Configuration with subset parameters
├── state_factory.py         # Target state preparation (unchanged)
├── noise_engine.py          # Noise application with subset twirling
├── amplified_swap.py        # SWAP test implementation (unchanged)
├── streaming_runner.py      # Main runner with subset-aware output
└── main_grid_run.py         # CLI with subset arguments

data/subsetTwirling_simulations/
└── steps_circuit_dephase_z_theta_phi_subset0.50.csv  # Example output
```

## Differences from Original Implementation

The original `moreNoise` implementation uses single-sample-per-copy twirling (one random Clifford per copy, averaging happens implicitly over many copies).

This `subsetTwirling` implementation:
1. **Explicitly averages** over a subset of Clifford combinations for each noise application
2. Allows **tunable computational cost** via subset_fraction
3. Provides **deterministic** results (when subset_mode="first_k")
4. Is more suitable for **iterative noise** scenarios where averaging must happen per iteration

## References

See manuscript Section on Clifford twirling for theoretical background.

## Contact

For questions or issues, contact the package maintainer or open an issue in the repository.
