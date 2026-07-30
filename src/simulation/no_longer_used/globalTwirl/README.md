# SWAP-Based Quantum Error Correction: Complete Technical Documentation

**Iterative Purification Simulation Framework**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Configuration System (configs.py)](#configuration-system-configspy)
4. [Target State Generation (state_factory.py)](#target-state-generation-state_factorypy)
5. [Quantum Noise Engine (noise_engine.py)](#quantum-noise-engine-noise_enginepy)
6. [SWAP Test & Purification (amplified_swap.py)](#swap-test--purification-amplified_swappy)
7. [Simulation Orchestration (streaming_runner.py)](#simulation-orchestration-streaming_runnerpy)
8. [Entry Point & Grid Execution (main_grid_run.py)](#entry-point--grid-execution-main_grid_runpy)
9. [System Integration & Data Flow](#system-integration--data-flow)
10. [CSV Output Specification](#csv-output-specification)
11. [Complete Execution Flow](#complete-execution-flow)

---

## Executive Summary

This document provides comprehensive technical documentation for the SWAP-based quantum error correction (QEC) simulation framework. The codebase implements iterative purification error correction using SWAP tests and amplitude amplification, with support for both regular streaming purification and iterative noise application modes.

The simulation framework consists of six core modules that work together to:

- Generate configurable target quantum states on M qubits
- Apply realistic noise models (depolarizing, dephasing) with optional Clifford twirling
- Perform SWAP-test purification with emulated amplitude amplification
- Support both single-shot and iterative purification protocols
- Generate comprehensive CSV datasets for analysis and visualization

---

## System Architecture Overview

The simulation framework follows a modular architecture with clear separation of concerns:

| Module                | Primary Responsibility       | Key Components                                                     |
| --------------------- | ---------------------------- | ------------------------------------------------------------------ |
| `configs.py`          | Configuration & Type System  | RunSpec, NoiseSpec, TargetSpec, parameter validation               |
| `state_factory.py`    | Target State Generation      | Hadamard, GHZ, Haar-random, manual states                          |
| `noise_engine.py`     | Noise Application & Twirling | Kraus operators, Clifford twirling, circuit & density matrix modes |
| `amplified_swap.py`   | SWAP Test & Purification     | Matrix SWAP unitary, amplitude amplification, state extraction     |
| `streaming_runner.py` | Simulation Orchestration     | Regular & iterative modes, streaming trees, metrics calculation    |
| `main_grid_run.py`    | Entry Point & Grid Sweeps    | CLI interface, parameter grids, batch execution                    |

---

## Configuration System (configs.py)

The configuration system provides a type-safe, hierarchical parameter management framework using Python dataclasses. All simulation parameters are centrally defined and validated through this module.

### Core Enumerations

#### NoiseType Enumeration

```python
class NoiseType(str, Enum):
    depolarizing = "depolarizing"    # Symmetric X,Y,Z mixing
    dephase_z = "dephase_z"          # Pure Z dephasing
    dephase_x = "dephase_x"          # Pure X dephasing
```

Defines the three supported quantum noise models:

- **`depolarizing`**: Isotropic mixing with X, Y, Z Pauli operators
- **`dephase_z`**: Pure Z-basis dephasing (computational basis)
- **`dephase_x`**: Pure X-basis dephasing (superposition basis)

#### NoiseMode Enumeration

```python
class NoiseMode(str, Enum):
    iid_p = "iid_p"        # Independent channel per qubit
    exact_k = "exact_k"    # Deterministic k-error injection
```

Controls how noise is applied to multi-qubit states:

- **`iid_p`**: Apply CPTP channel independently to each qubit with probability p
- **`exact_k`**: Inject exactly k single-qubit errors at deterministic locations

#### StateKind Enumeration

```python
class StateKind(str, Enum):
    manual = "manual"                  # User-provided circuit/vector
    haar = "haar"                      # Haar-random pure state
    random_circuit = "random_circuit"  # Shallow random rotations
    hadamard = "hadamard"              # Product |+⟩^⊗M
    ghz = "ghz"                        # Entangled (|00...0⟩ + |11...1⟩)/√2
```

### Parameter Conversion Functions

The module provides bidirectional conversion between physical error rates (δ) used in theoretical analysis and channel probabilities (p) used in Kraus operator implementations.

#### `delta_to_kraus_p(noise: NoiseType, delta: float) → float`

Converts manuscript physical error rate δ to channel probability p:

- **Depolarizing**: `p = 3δ/4` (since δ = 4p/3 in manuscript notation)
- **Dephasing**: `p = δ` (direct correspondence)

**Parameters:**

- `noise`: NoiseType enum value specifying the channel type
- `delta`: Physical error rate from theoretical analysis

**Returns:** Channel probability p for Kraus operator construction

#### `kraus_p_to_delta(noise: NoiseType, p: float) → float`

Inverse conversion from channel probability p to physical error rate δ.

**Parameters:**

- `noise`: NoiseType enum value
- `p`: Channel probability parameter

**Returns:** Physical error rate δ for theoretical comparison

### Configuration Dataclasses

#### TargetSpec Class

Specifies the target pure state |ψ⟩ for purification. All simulations aim to recover this reference state from noisy copies.

```python
@dataclass
class TargetSpec:
    M: int                                    # Number of qubits
    kind: StateKind = StateKind.haar          # State generation method
    manual_circuit: Optional[QuantumCircuit] = None
    manual_statevector: Optional[Statevector] = None
    random_layers: int = 3                    # Depth for random circuits
    seed: Optional[int] = None               # RNG seed for reproducibility
```

**Field Descriptions:**

- **`M`**: System size (number of qubits), determines Hilbert space dimension 2^M
- **`kind`**: Selects state preparation method from StateKind enum
- **`manual_circuit`**: Custom preparation circuit (used when kind=manual)
- **`manual_statevector`**: Direct state specification (used when kind=manual)
- **`random_layers`**: Circuit depth for random_circuit generation
- **`seed`**: Random number generator seed for deterministic state generation

#### NoiseSpec Class

Configures quantum noise application, including channel type, strength, and application mode.

```python
@dataclass
class NoiseSpec:
    noise_type: NoiseType = NoiseType.depolarizing
    mode: NoiseMode = NoiseMode.iid_p
    p: float = 0.1                           # Channel probability
    exact_k: int = 0                         # Error count for exact_k mode
```

**Key Methods:**

- **`kraus_p()`**: Returns channel probability p directly
- **`manuscript_delta()`**: Converts to manuscript δ notation for theoretical comparison

#### AASpec Class

Controls amplitude amplification parameters. The implementation uses emulated AA (computes required Grover iterations without applying them).

```python
@dataclass
class AASpec:
    target_success: float = 0.99           # Desired P[ancilla=0] after AA
    max_iters: int = 64                    # Maximum Grover iterations
    use_postselection_only: bool = False   # Skip AA, just postselect
```

#### TwirlingSpec Class

Configures Clifford twirling for dephasing noise mitigation. Automatically enabled for dephasing channels to convert anisotropic noise into effective depolarization.

```python
@dataclass
class TwirlingSpec:
    enabled: bool = True                    # Auto-enable for dephasing
    mode: Literal["random", "cyclic"] = "cyclic"  # Gate selection strategy
    seed: Optional[int] = None              # RNG seed for reproducibility
```

#### RunSpec Class

Top-level configuration container that aggregates all simulation parameters and provides validation and ID synthesis.

```python
@dataclass
class RunSpec:
    target: TargetSpec
    noise: NoiseSpec
    aa: AASpec
    twirling: TwirlingSpec = field(default_factory=TwirlingSpec)
    N: int = 16                             # Total copies (regular) or iterations parameter
    backend_method: str = "density_matrix"  # Qiskit backend
    out_dir: Path = "data/simulations"      # Output directory
    run_id: Optional[str] = None            # Custom identifier
    verbose: bool = False                   # Debug logging
    iterative_noise: bool = False           # Enable iterative mode
    purification_level: int = 1             # ℓ parameter (SWAP rounds per iteration)
```

**Critical Methods:**

##### `validate()`

Validates all parameters for consistency and bounds. Performs comprehensive checks:

- M value range (1 ≤ M ≤ 10 for practical memory limits)
- N value validity (must be power of 2 for streaming tree)
- Noise parameter bounds (0 ≤ p ≤ 1)
- exact_k consistency (k ≤ M)
- AA parameter validation
- Iterative mode parameter consistency

**Raises:** `ValueError` with descriptive message if validation fails

##### `synthesize_run_id()`

Generates unique identifier from parameters. Format:
`M{M}_N{N}_{noise_type}_{mode}_p{p:.5f}[_twirl][_iter_l{ℓ}]`

Example: `"M1_N64_depolarizing_iid_p_p0.50000_twirl"`

##### `_should_apply_twirling()`

Determines if Clifford twirling should be activated based on:

- `twirling.enabled` flag
- Noise type (auto-enable for dephasing channels)
- Mode compatibility (iid_p mode only)

**Returns:** `bool` indicating whether twirling should be applied

---

## Target State Generation (state_factory.py)

The state factory module generates preparation circuits and reference statevectors for target quantum states. It provides a unified interface for creating various types of M-qubit pure states used as purification targets.

### Core Function: build_target()

```python
def build_target(spec: TargetSpec) -> Tuple[QuantumCircuit, Statevector]:
```

Main entry point that constructs preparation circuit and reference statevector based on TargetSpec configuration.

**Parameters:**

- `spec`: TargetSpec object containing all target state configuration

**Returns:**

- `qc_prep`: QuantumCircuit that prepares |ψ⟩ from |00...0⟩
- `psi`: Reference Statevector(|ψ⟩) for fidelity calculations

**Implementation Flow:**

1. Dispatch to appropriate state generation method based on `spec.kind`
2. For `manual` kind: use provided circuit or statevector
3. For other kinds: call corresponding `_build_*()` helper function
4. Validate output state dimensions and normalization
5. Return preparation circuit and reference statevector

### State Generation Methods

#### Hadamard Product States

```python
def _build_hadamard(M: int) -> Tuple[QuantumCircuit, Statevector]:
```

Generates the M-qubit product state |+⟩^⊗M = (H|0⟩)^⊗M where |+⟩ = (|0⟩ + |1⟩)/√2.

**Parameters:**

- `M`: Number of qubits

**Implementation:**

1. Create QuantumCircuit with M qubits
2. Apply Hadamard gate to each qubit: `qc.h(i)` for i ∈ [0, M-1]
3. Compute exact statevector using Qiskit AerSimulator
4. Validate result has correct dimensions (2^M components)
5. Return (circuit, statevector) tuple

**Mathematical Result:** |ψ⟩ = 2^(-M/2) ∑\_{x∈{0,1}^M} |x⟩

**Advantages:** Efficient for testing due to separable structure, known analytical properties

#### GHZ States

```python
def _build_ghz(M: int) -> Tuple[QuantumCircuit, Statevector]:
```

Creates maximally entangled GHZ states: |GHZ_M⟩ = (|00...0⟩ + |11...1⟩)/√2.

**Parameters:**

- `M`: Number of qubits (M ≥ 2)

**Circuit Construction:**

1. Create QuantumCircuit with M qubits
2. Apply H gate to qubit 0: creates |+⟩ = (|0⟩ + |1⟩)/√2
3. For i = 1, ..., M-1: apply CNOT(0, i) to entangle all qubits
4. Result: |0⟩⊗|+⟩ + |1⟩⊗|+⟩ → |00...0⟩ + |11...1⟩

**Mathematical Result:** |ψ⟩ = 2^(-1/2) (|0⟩^⊗M + |1⟩^⊗M)

**Properties:** Maximally entangled, symmetric under qubit permutation, sensitive to any single-qubit error

#### Haar-Random States

```python
def _build_haar(M: int, seed: Optional[int]) -> Tuple[QuantumCircuit, Statevector]:
```

Generates uniformly random pure states from the Haar measure on the 2^M-dimensional Hilbert space.

**Parameters:**

- `M`: Number of qubits
- `seed`: Random number generator seed for reproducibility

**Algorithm:**

1. Set random seed if provided: `np.random.seed(seed)`
2. Sample 2^M complex coefficients from normal distribution:
   ```python
   real_parts = np.random.normal(0, 1, 2**M)
   imag_parts = np.random.normal(0, 1, 2**M)
   coeffs = real_parts + 1j * imag_parts
   ```
3. Normalize to unit length: `ψ = coeffs / ||coeffs||_2`
4. Use Qiskit Initialize gate to synthesize preparation circuit
5. Circuit depth scales exponentially with M

**Mathematical Result:** |ψ⟩ uniformly distributed according to Haar measure on pure states

**Use Cases:** Testing with typical entangled states, worst-case performance analysis

#### Random Circuit States

```python
def _build_random_circuit(M: int, layers: int, seed: Optional[int]) -> Tuple[QuantumCircuit, Statevector]:
```

Creates pseudo-random states using shallow parameterized circuits. Provides controllable entanglement without exponential synthesis cost.

**Parameters:**

- `M`: Number of qubits
- `layers`: Number of circuit layers to apply
- `seed`: Random number generator seed

**Layer Structure (repeated `layers` times):**

1. **Single-qubit layer**: Apply RX(θ), RY(φ), RZ(χ) to each qubit with random angles

   ```python
   for i in range(M):
       theta = np.random.uniform(0, 2*np.pi)
       phi = np.random.uniform(0, 2*np.pi)
       chi = np.random.uniform(0, 2*np.pi)
       qc.rx(theta, i)
       qc.ry(phi, i)
       qc.rz(chi, i)
   ```

2. **Entangling layer**: Apply CZ gates in ring topology (i ↔ (i+1) mod M)
   ```python
   for i in range(M):
       qc.cz(i, (i + 1) % M)
   ```

**Result:** Polynomial-depth circuit with controlled entanglement structure, suitable for testing purification protocols without exponential overhead.

---

## Quantum Noise Engine (noise_engine.py)

The noise engine implements realistic quantum decoherence models using explicit Kraus operator formulations. It supports both circuit-based noise (for Qiskit simulation) and direct density matrix operations (for iterative purification). The module implements channel twirling per manuscript Equation 54 for dephasing noise mitigation.

### Kraus Operator Construction

All noise models are implemented as explicit Kraus operators rather than using Qiskit's built-in channels, providing full control over the mathematical form and ensuring consistency with theoretical analysis.

#### Depolarizing Channel

```python
def _kraus_depolarizing(p: float) -> Kraus:
```

Implements single-qubit depolarizing channel with Kraus operators.

**Parameters:**

- `p`: Channel probability parameter (0 ≤ p ≤ 1)

**Kraus Operators:**

- E₀ = √(1-p) I (identity with probability 1-p)
- E₁ = √(p/3) X (X error with probability p/3)
- E₂ = √(p/3) Y (Y error with probability p/3)
- E₃ = √(p/3) Z (Z error with probability p/3)

**Channel Action:** ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

**Manuscript Relation:** δ = 4p/3, so physical error rate δ corresponds to channel parameter p = 3δ/4.

**Implementation:**

```python
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

kraus_ops = [
    np.sqrt(1 - p) * I,
    np.sqrt(p/3) * X,
    np.sqrt(p/3) * Y,
    np.sqrt(p/3) * Z
]
return Kraus(kraus_ops)
```

#### Pure Dephasing Channels

```python
def _kraus_z_dephase(p: float) -> Kraus:
def _kraus_x_dephase(p: float) -> Kraus:
```

**Z-dephasing (computational basis decoherence):**

- E₀ = √(1-p) I
- E₁ = √p Z
- Action: ρ → (1-p)ρ + p ZρZ

**X-dephasing (superposition basis decoherence):**

- E₀ = √(1-p) I
- E₁ = √p X
- Action: ρ → (1-p)ρ + p XρX

**Manuscript Relation:** For pure dephasing channels, δ = p (direct correspondence between physical and channel parameters).

### Clifford Twirling Implementation

Clifford twirling converts anisotropic dephasing noise into effective isotropic depolarization by averaging over single-qubit Clifford conjugations. This implements the channel twirling protocol from manuscript Equation 54.

#### Clifford Gate Set

```python
def _sample_clifford_gate(mode: str, index: int, seed: Optional[int] = None) -> str:
```

Samples from the three-element Clifford set {I, H, SH} that maps Z to {Z, X, Y} respectively.

**Parameters:**

- `mode`: Sampling strategy ("random" or "cyclic")
- `index`: Deterministic index for cyclic mode
- `seed`: Random seed for random mode

**Gate Set:**

- `'i'`: Identity gate (Z → Z)
- `'h'`: Hadamard gate (Z → X)
- `'sh'`: S·H composition (Z → Y)

**Sampling Modes:**

- **`random`**: Uniform random selection with optional seed
- **`cyclic`**: Deterministic cycling based on index: gates[index % 3]

**Returns:** String identifier for the selected Clifford gate

#### Channel Twirling Protocol

For dephasing noise with Clifford twirling enabled, the protocol is:

1. Apply random single-qubit Clifford C to |ψ⟩ → |ψ'⟩ = C|ψ⟩
2. Apply dephasing channel in rotated frame
3. Apply inverse Clifford C† to return to original frame

**Result:** ρ_twirled = C† E_deph(C ρ C†) C

This protocol averages Z-dephasing over the orbit {Z, X, Y}, producing effective depolarization noise that is more amenable to error correction.

### Noise Application Modes

#### Circuit-Based Noise (IID Mode)

```python
def build_copy_iid_p(prep: QuantumCircuit, noise: NoiseSpec,
                     twirling: Optional[TwirlingSpec] = None,
                     twirl_seed: Optional[int] = None) -> QuantumCircuit:
```

Creates a quantum circuit that prepares a noisy copy by appending noise channels to the target preparation circuit.

**Parameters:**

- `prep`: Target preparation circuit
- `noise`: NoiseSpec configuration
- `twirling`: Optional TwirlingSpec for Clifford twirling
- `twirl_seed`: Random seed for twirling gate selection

**Construction Steps:**

1. Start with target preparation circuit: `qc = prep.copy()`
2. If twirling enabled: append random Cliffords to all qubits
3. Append noise channel to each qubit independently
4. If twirling enabled: append inverse Cliffords

**Implementation Details:**

- Uses Qiskit's built-in noise channels for circuit simulation
- Handles twirling by bracketing noise with Clifford gates
- Returns circuit implementing ⊗ᵢ Eᵢ(·) where Eᵢ is the single-qubit channel applied to qubit i

#### Direct Density Matrix Noise

```python
def apply_noise_to_density_matrix(rho: DensityMatrix, noise: NoiseSpec,
                                  twirling: Optional[TwirlingSpec] = None,
                                  twirl_seed: Optional[int] = None) -> DensityMatrix:
```

Applies noise directly to density matrices without circuit simulation. Critical for iterative purification where noise must be applied to intermediate states.

**Parameters:**

- `rho`: Input density matrix (M-qubit system)
- `noise`: NoiseSpec configuration
- `twirling`: Optional TwirlingSpec
- `twirl_seed`: Random seed for twirling

**Implementation Flow:**

1. Extract system size M from density matrix dimensions
2. If twirling enabled: apply frame rotation
3. Apply noise channel to each qubit sequentially
4. If twirling enabled: apply inverse frame rotation
5. Return modified density matrix

**Core Implementation:**

```python
# Apply noise to each qubit
for qubit_idx in range(M):
    # Get single-qubit Kraus operators
    kraus = _get_kraus_operators(noise.noise_type, noise.p)

    # Extend to full system
    for E in kraus.data:
        E_full = _single_qubit_to_full_operator(E, qubit_idx, M)
        rho_new += E_full @ rho.data @ E_full.conj().T
```

**Helper Function: `_single_qubit_to_full_operator()`**

```python
def _single_qubit_to_full_operator(single_op: np.ndarray, target_qubit: int, M: int) -> np.ndarray:
```

Extends single-qubit operator to full M-qubit system using tensor products:
`E_full = I ⊗ ... ⊗ E ⊗ ... ⊗ I`

#### Exact-k Error Injection

```python
def sample_error_pattern(M: int, noise_type: NoiseType, k: int,
                         seed: Optional[int] = None) -> ErrorPattern:
```

For exact_k mode, generates deterministic error patterns with exactly k single-qubit Pauli faults.

**Parameters:**

- `M`: Number of qubits
- `noise_type`: Type of errors to inject
- `k`: Exact number of errors (k ≤ M)
- `seed`: Random seed for reproducible patterns

**Error Pattern Selection:**

1. Randomly select k distinct qubits (no replacement)
2. For each selected qubit, choose error type:
   - **Depolarizing**: Uniform selection from {X, Y, Z}
   - **Z-dephasing**: Only Z errors
   - **X-dephasing**: Only X errors

**Returns:** ErrorPattern object containing:

- `error_qubits`: List of qubit indices with errors
- `error_types`: List of Pauli operators for each error

This mode is essential for SWAP test theory compliance, as it ensures identical error patterns can be applied to both copies entering each merge operation.

---

## SWAP Test & Purification (amplified_swap.py)

This module implements the core SWAP-test quantum purification protocol with emulated amplitude amplification. It constructs explicit unitary matrices to avoid Qiskit qubit ordering inconsistencies and provides precise control over the purification process.

### SWAP Test Unitary Construction

#### Matrix-Based Implementation

```python
def build_swap_test_unitary(M: int) -> np.ndarray:
```

Constructs the (1+2M)-qubit SWAP test unitary as an explicit matrix to avoid Qiskit's qubit ordering conventions.

**Parameters:**

- `M`: Number of qubits per register

**System Layout:** |anc⟩|A₀...A*{M-1}⟩|B₀...B*{M-1}⟩

**Construction Process:**

1. **Build H_anc = H ⊗ I^{2M}** (Hadamard on ancilla, identity on registers)

   ```python
   H_single = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   I_2M = np.eye(2**(2*M))
   H_anc = np.kron(H_single, I_2M)
   ```

2. **Build controlled-SWAP matrix** (controlled by ancilla)

   ```python
   total_dim = 2**(1 + 2*M)
   cswap = np.eye(total_dim, dtype=complex)

   for idx in range(total_dim):
       anc_bit = (idx >> (2*M)) & 1
       if anc_bit == 1:
           # Extract A and B register indices
           A_idx = (idx >> M) & ((1 << M) - 1)
           B_idx = idx & ((1 << M) - 1)

           # Build swapped index
           swapped_idx = (1 << (2*M)) | (B_idx << M) | A_idx

           # Swap matrix rows
           cswap[idx, idx] = 0
           cswap[swapped_idx, idx] = 1
   ```

3. **Compose final unitary**: U_swap = H_anc × CSWAP × H_anc

**Key Implementation Details:**

- Uses bit manipulation to handle qubit indexing consistently
- When ancilla = 0: do nothing
- When ancilla = 1: swap each A_i ↔ B_i
- Explicit matrix construction eliminates Qiskit version dependencies

#### Controlled-SWAP Implementation Details

The controlled-SWAP gate operates on computational basis states by examining each basis state |anc, A, B⟩:

**Bit Extraction:**

- Ancilla bit: `anc_bit = (idx >> (2*M)) & 1`
- A register: `A_idx = (idx >> M) & ((1 << M) - 1)`
- B register: `B_idx = idx & ((1 << M) - 1)`

**Conditional Swapping:**

- If `anc_bit = 1`: swap matrix rows corresponding to |1,A,B⟩ ↔ |1,B,A⟩
- If `anc_bit = 0`: leave unchanged

### Amplitude Amplification (Emulated)

The implementation uses emulated amplitude amplification: it computes the required number of Grover iterations but applies only the SWAP test unitary. The conditional output state given ancilla = 0 is identical whether amplitude amplification was applied or not.

#### Success Probability Calculation

```python
def ancilla_success_probability(rho_after_swap: DensityMatrix, M: int) -> float:
```

Computes P[ancilla = 0] from the (1+2M)-qubit density matrix after SWAP test application.

**Parameters:**

- `rho_after_swap`: Joint density matrix on ancilla + 2 registers
- `M`: Number of qubits per register

**Implementation:**

```python
# Build projector Π₀ = |0⟩⟨0|_anc ⊗ I_A ⊗ I_B
total_dim = 2**(1 + 2*M)
proj_matrix = np.zeros((total_dim, total_dim), dtype=complex)

# Project onto ancilla = 0 subspace
for idx in range(total_dim):
    anc_bit = (idx >> (2*M)) & 1
    if anc_bit == 0:
        proj_matrix[idx, idx] = 1.0

# Compute success probability
P0 = np.real(np.trace(proj_matrix @ rho_after_swap.data))
```

**Critical Implementation Note:** This computes Tr(Π₀ ρ) where Π₀ = |0⟩⟨0|\_anc ⊗ I_A ⊗ I_B. For controlled gates, projection must occur before partial tracing, not after.

#### Grover Iteration Selection

```python
def choose_grover_iters(P0: float, target_success: float, max_iters: int) -> int:
```

Selects optimal number of Grover iterations k to achieve target success probability.

**Parameters:**

- `P0`: Initial success probability from SWAP test
- `target_success`: Desired success probability (typically 0.99)
- `max_iters`: Maximum allowed iterations

**Algorithm:**

1. **Compute rotation angle**: `θ = 2 * arcsin(√P₀)`
2. **Find optimal k** minimizing `|sin²((k+½)θ) - target_success|`
3. **Analytical approximation**: `k ≈ π/(2θ) - ½` for maximum amplification
4. **Apply bounds**: `k ∈ [0, max_iters]`

**Edge Cases:**

- If P₀ ≥ target_success: return k = 0 (no amplification needed)
- If P₀ ≤ 0 or P₀ ≥ 1: handle gracefully to avoid mathematical errors

**Implementation:**

```python
if P0 >= target_success:
    return 0

if P0 <= 0.0 or P0 >= 1.0:
    return 0

theta = 2 * np.arcsin(np.sqrt(P0))
if theta <= 0:
    return 0

# Find optimal k
optimal_k = int(np.round(np.pi / (2 * theta) - 0.5))
return max(0, min(optimal_k, max_iters))
```

### State Projection and Extraction

#### Ancilla Projection

```python
def _project_ancilla_zero(rho: DensityMatrix, M: int) -> DensityMatrix:
```

Projects the ancilla subsystem onto |0⟩⟨0| and renormalizes the resulting state.

**Parameters:**

- `rho`: Joint density matrix on (1+2M) qubits
- `M`: Number of qubits per register

**Implementation Steps:**

1. **Build projector** Π₀ = |0⟩⟨0|\_anc ⊗ I_A ⊗ I_B
2. **Apply projection**: ρ' = Π₀ ρ Π₀†
3. **Compute normalization**: p₀ = Tr(ρ')
4. **Return normalized state**: ρ'/p₀

```python
total_dim = 2**(1 + 2*M)
proj_matrix = np.zeros((total_dim, total_dim), dtype=complex)

for idx in range(total_dim):
    anc_bit = (idx >> (2*M)) & 1
    if anc_bit == 0:
        proj_matrix[idx, idx] = 1.0

# Project and normalize
rho_proj = proj_matrix @ rho.data @ proj_matrix.conj().T
norm = np.real(np.trace(rho_proj))

if norm > 1e-15:
    rho_normalized = rho_proj / norm
    return DensityMatrix(rho_normalized)
else:
    # Fallback for numerical issues
    return DensityMatrix(np.eye(total_dim) / total_dim)
```

#### Purified State Extraction

```python
def extract_purified_register(rho_after_proj: DensityMatrix, M: int) -> DensityMatrix:
```

Extracts the purified single-register state by tracing out the ancilla and register B.

**Parameters:**

- `rho_after_proj`: Density matrix after ancilla projection
- `M`: Number of qubits per register

**Subsystem Tracing Order:**

1. **Trace out register B** (qubits M+1 through 2M): ρ_anc,A = Tr_B(ρ)
2. **Trace out ancilla** (qubit 0): ρ_A = Tr_anc(ρ_anc,A)

**Implementation:**

```python
# First trace out register B
dims = [2] * (1 + 2*M)  # [anc, A0, A1, ..., A_{M-1}, B0, B1, ..., B_{M-1}]
subsystems_to_trace = list(range(1+M, 1+2*M))  # Register B indices

rho_anc_A = partial_trace(rho_after_proj, subsystems_to_trace)

# Then trace out ancilla
rho_A = partial_trace(rho_anc_A, [0])  # Ancilla index

return DensityMatrix(rho_A.data)
```

### Main Purification Interface

```python
def purify_two_from_density(rho_A: DensityMatrix, rho_B: DensityMatrix,
                            aa: AASpec) -> Tuple[DensityMatrix, Dict]:
```

Main entry point for SWAP-test purification of two M-qubit density matrices.

**Parameters:**

- `rho_A`: First input density matrix
- `rho_B`: Second input density matrix
- `aa`: AASpec configuration for amplitude amplification

**Critical Requirement:** For SWAP test theory to hold, rho_A and rho_B must be identical copies (modulo independent noise realizations).

**Purification Protocol:**

1. **Construct joint state**: |0⟩⟨0|\_anc ⊗ ρ_A ⊗ ρ_B

   ```python
   M = int(np.log2(rho_A.data.shape[0]))

   # Build joint initial state
   anc_zero = np.array([[1, 0], [0, 0]], dtype=complex)
   rho_joint = np.kron(anc_zero, np.kron(rho_A.data, rho_B.data))
   ```

2. **Apply SWAP test unitary**: ρ' = U_swap ρ U_swap†

   ```python
   U_swap = build_swap_test_unitary(M)
   rho_after_swap = U_swap @ rho_joint @ U_swap.conj().T
   ```

3. **Calculate success probability and optimal Grover iterations**

   ```python
   P0 = ancilla_success_probability(DensityMatrix(rho_after_swap), M)
   k = choose_grover_iters(P0, aa.target_success, aa.max_iters)
   ```

4. **Project ancilla to |0⟩ and extract register A**
   ```python
   rho_projected = _project_ancilla_zero(DensityMatrix(rho_after_swap), M)
   rho_purified = extract_purified_register(rho_projected, M)
   ```

**Returns:**

- `rho_out`: Purified M-qubit density matrix
- `metrics`: Dictionary with `P_success` and `grover_iters`

---

## Simulation Orchestration (streaming_runner.py)

The streaming runner is the central orchestration module that implements both regular streaming purification and iterative purification protocols. It manages the entire simulation workflow from state preparation through noise application to purification and metrics collection.

### Execution Mode Dispatch

```python
def run_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
```

Main entry point that dispatches to either regular streaming or iterative purification based on the `spec.iterative_noise` flag.

**Parameters:**

- `spec`: RunSpec containing complete simulation configuration

**Mode Selection:**

- `spec.iterative_noise = False`: Call `run_regular_streaming()`
- `spec.iterative_noise = True`: Call `run_iterative_purification()`

**Returns:**

- `steps_df`: DataFrame with step-by-step purification data
- `finals_df`: DataFrame with summary metrics

### Iterative Purification (Option B Protocol)

The iterative purification mode implements the "Option B" protocol with deterministic global Clifford cycling for theoretical rigor.

#### Protocol Structure

The iterative protocol follows this structure per iteration t:

1. **Global Frame Rotation**: Apply global Clifford C_t to ALL qubits
2. **Noise Application**: Apply noise channel once to current state
3. **Frame Restoration**: Apply C_t† to return to original frame
4. **Identical Copy Creation**: Clone 2^ℓ identical copies of noisy state
5. **Clean Purification**: Apply ℓ levels of SWAP purification

#### Global Clifford Cycling

```python
def _cycle_gate_for_iteration(iter_idx: int) -> str:
    """Deterministic cycling through Clifford gates."""
    cycle = ["i", "h", "sh"]
    return cycle[iter_idx % len(cycle)]

def _U_global(gate: str, M: int) -> np.ndarray:
    """Build global Clifford unitary."""
    if gate == "i":
        return np.eye(2**M, dtype=complex)
    elif gate == "h":
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    elif gate == "sh":
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        SH = S @ H
        gate_matrix = SH

    # Build tensor product: (U_single)^⊗M
    result = np.array([[1.0]], dtype=complex)
    for _ in range(M):
        result = np.kron(result, gate_matrix)
    return result
```

Implements deterministic cycling through the three-element set {I, H, SH}:

- **Iteration 0**: Identity gate (I) - no rotation
- **Iteration 1**: Hadamard gate (H) - Z basis → X basis
- **Iteration 2**: H·S composition - Z basis → Y basis
- **Iteration 3**: Cycle repeats (back to I)

The global unitary is constructed as U_global = (U_single)^⊗M where U_single is the chosen single-qubit Clifford gate.

#### Parameter Mapping

The iterative mode repurposes the N parameter to control iteration count:

- **`num_iterations`**: Number of noise→purification cycles = log₂(N)
- **`purification_level (ℓ)`**: SWAP rounds per cycle
- **`num_copies_needed`**: 2^ℓ copies required per iteration

**Example Mappings:**

- N=16 → 4 iterations, ℓ=2 → 4 copies per iteration, 8 total SWAP rounds
- N=64 → 6 iterations, ℓ=3 → 8 copies per iteration, 18 total SWAP rounds

#### Main Iterative Function

```python
def run_iterative_purification(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
```

**Implementation Flow:**

1. **Setup and Validation**

   ```python
   M = spec.target.M
   num_iterations = int(np.log2(spec.N))
   purification_level = spec.purification_level  # ℓ
   num_copies_needed = 2 ** purification_level

   if 2 ** num_iterations != spec.N:
       raise ValueError(f"N must be power of 2 for iterative mode, got {spec.N}")
   ```

2. **Target State Preparation**

   ```python
   qc_prep, psi_target = build_target(spec.target)
   current_state = Statevector(psi_target.data.copy()).to_density_matrix()
   ```

3. **Iteration Loop**

   ```python
   for iter_idx in range(num_iterations):
       # Step 1: Apply global frame rotation
       gate = _cycle_gate_for_iteration(iter_idx)
       U_global = _U_global(gate, M)

       # Rotate state: |ψ⟩ → U|ψ⟩
       current_state_data = current_state.data
       rotated_data = U_global @ current_state_data @ U_global.conj().T
       rotated_state = DensityMatrix(rotated_data)

       # Step 2: Apply noise in rotated frame
       noisy_state = apply_noise_to_density_matrix(
           rotated_state, spec.noise,
           twirling=None  # No additional twirling
       )

       # Step 3: Restore original frame
       restored_data = U_global.conj().T @ noisy_state.data @ U_global
       final_noisy_state = DensityMatrix(restored_data)

       # Step 4: Create identical copies
       noisy_copies = [
           DensityMatrix(final_noisy_state.data.copy())
           for _ in range(num_copies_needed)
       ]

       # Step 5: Apply streaming purification
       current_state, step_records = _run_streaming_tree_on_copies(
           noisy_copies, spec.aa, iter_idx, spec
       )
   ```

#### Streaming Tree Logic

Within each iteration, the 2^ℓ identical noisy copies are processed through a binary tree of SWAP purification operations:

```python
def _run_streaming_tree_on_copies(copies: List[DensityMatrix], aa: AASpec,
                                  iteration: int, spec: RunSpec) -> Tuple[DensityMatrix, List]:
    """Process copies through streaming binary tree."""
    M = spec.target.M
    slots: Dict[int, DensityMatrix] = {}  # level -> density matrix
    step_records = []

    for copy_idx, noisy_copy in enumerate(copies):
        level = 0
        carry_dm = noisy_copy

        while True:
            if level not in slots:
                # Store at empty level
                slots[level] = carry_dm
                break
            else:
                # Retrieve waiting copy and merge
                left = slots.pop(level)
                purified_state, meta = purify_two_from_density(left, carry_dm, aa)

                # Record metrics for this merge
                step_record = _build_step_record(
                    purified_state, level, iteration, meta, spec
                )
                step_records.append(step_record)

                carry_dm = purified_state
                level += 1

    # Final state is at highest level
    max_level = max(slots.keys())
    final_state = slots[max_level]

    return final_state, step_records
```

This implements a streaming binary tree where copies are merged as soon as a pair is available at each level. The final result emerges at level ℓ.

### Regular Streaming Purification

```python
def run_regular_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
```

Implements the original streaming purification algorithm where N independent noisy copies are generated and progressively merged through a logarithmic-depth tree.

#### Copy Generation Strategies

The module supports multiple copy generation strategies:

**IID Cached Mode:**

- Used when: `mode = iid_p AND twirling = False`
- Generate single noisy circuit, reuse for all copies
- Maximum efficiency for identical copies

```python
if spec.noise.mode == NoiseMode.iid_p and not should_apply_twirling:
    # Generate single cached circuit
    cached_circuit = build_noisy_copy(qc_prep, spec.noise, None, seed=0)

    # Reuse for all copies
    for copy_idx in range(spec.N):
        result = execute_circuit(cached_circuit)
        dm = result.density_matrix()
        copies.append(dm)
```

**Independent Generation Mode:**

- Used when: twirling enabled OR mode = exact_k
- Generate fresh circuit for each copy with different random seeds
- Supports twirling randomization and exact_k error patterns

```python
for copy_idx in range(spec.N):
    copy_seed = base_seed + copy_idx if base_seed else None

    if spec.noise.mode == NoiseMode.exact_k:
        error_pattern = sample_error_pattern(M, spec.noise.noise_type,
                                           spec.noise.exact_k, copy_seed)
        qc_noisy = build_noisy_copy(qc_prep, spec.noise, error_pattern)
    else:
        qc_noisy = build_noisy_copy(qc_prep, spec.noise, None, copy_seed)

    result = execute_circuit(qc_noisy)
    dm = result.density_matrix()
    copies.append(dm)
```

#### Exact-k Identical Pairs

For exact_k mode at level 0 merges, the code enforces identical error patterns:

```python
if spec.noise.mode == NoiseMode.exact_k and level == 0:
    # Use shared seed for identical error patterns
    shared_seed = merge_counter * 1000 + level
    shared_pattern = sample_error_pattern(M, spec.noise.noise_type,
                                         spec.noise.exact_k, shared_seed)

    qc_left = build_noisy_copy(qc_prep, spec.noise, shared_pattern=shared_pattern)
    qc_right = build_noisy_copy(qc_prep, spec.noise, shared_pattern=shared_pattern)

    dm_left = execute_circuit(qc_left).density_matrix()
    dm_right = execute_circuit(qc_right).density_matrix()
else:
    # Independent generation for higher levels
    dm_left = slots.pop(level)
    dm_right = current_copy
```

This ensures SWAP test theory compliance by guaranteeing identical inputs at the first purification level.

#### Regular Streaming Algorithm

```python
def run_regular_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main regular streaming implementation."""

    # Generate all noisy copies
    noisy_copies = generate_noisy_copies(spec)

    # Process through streaming tree
    slots: Dict[int, DensityMatrix] = {}
    step_records = []
    merge_counter = 0

    for copy_idx, noisy_copy in enumerate(noisy_copies):
        level = 0
        carry_dm = noisy_copy

        while True:
            if level not in slots:
                slots[level] = carry_dm
                break
            else:
                left = slots.pop(level)
                purified_state, meta = purify_two_from_density(left, carry_dm, spec.aa)

                # Calculate metrics and record
                step_record = {
                    'merge_num': merge_counter,
                    'level': level,
                    'copies_used': 2 ** (level + 1),
                    'fidelity': _fidelity_to_pure(purified_state, psi_target),
                    'eps_L': _trace_distance_to_pure(purified_state, psi_target),
                    'purity': _purity(purified_state),
                    'P_success': meta['P_success'],
                    'grover_iters': meta['grover_iters'],
                    # ... additional metrics
                }
                step_records.append(step_record)

                carry_dm = purified_state
                level += 1
                merge_counter += 1

    # Build final summary
    final_state = slots[max(slots.keys())]
    finals_record = _build_finals_record(final_state, step_records[0], spec)

    return pd.DataFrame(step_records), pd.DataFrame([finals_record])
```

### Metrics Collection and Output

#### Quantum State Metrics

The module computes comprehensive quantum state characterization metrics:

```python
def _fidelity_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """Compute fidelity F = ⟨ψ|ρ|ψ⟩."""
    return np.real(np.conj(psi.data) @ rho.data @ psi.data)

def _trace_distance_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """Compute trace distance ε_L = ½||ρ - |ψ⟩⟨ψ|||₁."""
    psi_dm = np.outer(psi.data, np.conj(psi.data))
    diff = rho.data - psi_dm
    eigenvals = np.linalg.eigvals(diff)
    return 0.5 * np.sum(np.abs(np.real(eigenvals)))

def _purity(rho: DensityMatrix) -> float:
    """Compute purity Tr(ρ²)."""
    return np.real(np.trace(rho.data @ rho.data))

def _bloch_vector_magnitude(rho: DensityMatrix) -> Optional[float]:
    """Compute |r⃗| for single qubits where ρ = (I + r⃗·σ⃗)/2."""
    if rho.data.shape != (2, 2):
        return None

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Extract Bloch vector components
    r_x = np.real(np.trace(rho.data @ sigma_x))
    r_y = np.real(np.trace(rho.data @ sigma_y))
    r_z = np.real(np.trace(rho.data @ sigma_z))

    return np.sqrt(r_x**2 + r_y**2 + r_z**2)
```

**Metric Definitions:**

- **Fidelity**: F = ⟨ψ|ρ|ψ⟩ (overlap with target pure state)
- **Trace Distance**: ε_L = (1/2)||ρ - |ψ⟩⟨ψ|||₁ (error rate)
- **Purity**: Tr(ρ²) (mixedness quantification, 1 for pure states)
- **Bloch Magnitude**: |r⃗| for single qubits where ρ = (I + r⃗·σ⃗)/2

#### Record Building Functions

```python
def _build_step_record(state: DensityMatrix, level: int, iteration: Optional[int],
                       meta: Dict, spec: RunSpec) -> Dict:
    """Build step record for CSV output."""

    base_record = {
        'run_id': spec.synthesize_run_id(),
        'M': spec.target.M,
        'depth': level if iteration is None else iteration * spec.purification_level + level,
        'noise': spec.noise.noise_type.value,
        'mode': spec.noise.mode.value,
        'p': spec.noise.p,
        'p_channel': spec.noise.p,  # Backward compatibility
        'fidelity': _fidelity_to_pure(state, psi_target),
        'eps_L': _trace_distance_to_pure(state, psi_target),
        'purity': _purity(state),
        'bloch_r': _bloch_vector_magnitude(state),
        'P_success': meta.get('P_success', 0.0),
        'grover_iters': meta.get('grover_iters', 0),
        'twirling_applied': spec._should_apply_twirling(),
    }

    # Add iterative-specific fields
    if iteration is not None:
        base_record.update({
            'iteration': iteration + 1,  # 1-indexed for output
            'purification_level': spec.purification_level,
            'merge_num': iteration + 1,  # Use iteration as merge number
        })

    return base_record
```

#### CSV Output Structure

The simulation generates two CSV files with comprehensive data:

**Steps CSV (step-by-step data):**

- One row per purification merge or iteration
- Contains intermediate metrics at each purification level
- Includes iteration and purification_level columns for iterative mode

**Finals CSV (summary data):**

- One row per complete run
- Initial vs final state comparison
- Error reduction ratios and aggregate statistics

---

## Entry Point & Grid Execution (main_grid_run.py)

The main grid runner provides the command-line interface and orchestrates large-scale parameter sweeps. It handles argument parsing, parameter grid generation, batch execution, and error recovery for comprehensive simulation campaigns.

### Command-Line Interface

The CLI supports flexible parameter specification and execution modes:

```bash
python -m src.simulation.main_grid_run [options]

Key arguments:
  --out DIR                    Output directory for CSV files
  --max-m INT                  Maximum M value (≤6 recommended)
  --m-values INT [INT ...]     Specific M values to test
  --noise {all,depol,z,x}      Noise type selection
  --mode {iid_p,exact_k}       Noise application mode
  --no-twirl                   Disable Clifford twirling
  --iterative                  Enable iterative purification mode
  --purification-level INT     ℓ parameter for iterative mode
  --quick                      Reduced parameter space for testing
  --verbose                    Enable debug logging
```

### Argument Parsing Implementation

```python
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SWAP purification grid simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full grid sweep
  python -m main_grid_run --out results/full_sweep

  # Quick test with specific M values
  python -m main_grid_run --m-values 1 2 --quick --out results/test

  # Iterative mode with ℓ=2
  python -m main_grid_run --iterative --purification-level 2 --out results/iter

  # Depolarizing noise only, no twirling
  python -m main_grid_run --noise depol --no-twirl --out results/depol_only
        """
    )

    parser.add_argument('--out', type=str, default='data/simulations_moreNoise',
                       help='Output directory for CSV files')
    parser.add_argument('--max-m', type=int, default=6,
                       help='Maximum M value (default: 6)')
    parser.add_argument('--m-values', type=int, nargs='*',
                       help='Specific M values to test (overrides --max-m)')
    parser.add_argument('--noise', choices=['all', 'depol', 'z', 'x'],
                       default='all', help='Noise types to simulate')
    parser.add_argument('--mode', choices=['iid_p', 'exact_k'],
                       default='iid_p', help='Noise application mode')
    parser.add_argument('--no-twirl', action='store_true',
                       help='Disable Clifford twirling')
    parser.add_argument('--iterative', action='store_true',
                       help='Enable iterative purification mode')
    parser.add_argument('--purification-level', type=int, default=1,
                       help='ℓ parameter for iterative mode (default: 1)')
    parser.add_argument('--quick', action='store_true',
                       help='Use reduced parameter grid for testing')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable debug logging')

    return parser.parse_args()
```

### Parameter Grid Configuration

Default parameter grids are defined as module constants:

```python
# System sizes (limited by memory for density matrix simulation)
M_LIST = [1, 5]

# Copy counts for regular mode / iteration parameters for iterative mode
N_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Error rates (channel probabilities)
P_LIST = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Noise types
NOISES = [NoiseType.depolarizing, NoiseType.dephase_z]

# Quick mode (reduced grid for testing)
N_LIST_QUICK = [4, 16, 64]
P_LIST_QUICK = [0.1, 0.5, 0.9]
```

**Grid Size Calculations:**

- **Full grid**: 2 M values × 10 N values × 10 p values × 2 noise types = 400 runs
- **Quick mode**: 2 M values × 3 N values × 3 p values × 2 noise types = 36 runs
- **M values capped at 6** for density matrix practicality (2^M scaling)

### Grid Generation Function

```python
def generate_parameter_grid(args: argparse.Namespace) -> List[Tuple]:
    """Generate Cartesian product of all parameter combinations."""

    # Determine M values
    if args.m_values:
        m_list = [m for m in args.m_values if 1 <= m <= 10]
    else:
        m_list = [m for m in M_LIST if m <= args.max_m]

    # Select parameter lists based on quick mode
    n_list = N_LIST_QUICK if args.quick else N_LIST
    p_list = P_LIST_QUICK if args.quick else P_LIST

    # Determine noise types
    if args.noise == 'all':
        noise_list = NOISES
    elif args.noise == 'depol':
        noise_list = [NoiseType.depolarizing]
    elif args.noise == 'z':
        noise_list = [NoiseType.dephase_z]
    elif args.noise == 'x':
        noise_list = [NoiseType.dephase_x]

    # Generate Cartesian product
    param_combinations = []
    for M in m_list:
        for N in n_list:
            for p in p_list:
                for noise_type in noise_list:
                    param_combinations.append((M, N, p, noise_type))

    return param_combinations
```

### RunSpec Generation

```python
def build_run_spec(M: int, N: int, p: float, noise_type: NoiseType,
                   args: argparse.Namespace) -> RunSpec:
    """Build RunSpec from parameter combination and CLI arguments."""

    # Base configuration
    target_spec = TargetSpec(M=M, kind=StateKind.hadamard, seed=42)
    noise_spec = NoiseSpec(
        noise_type=noise_type,
        mode=NoiseMode[args.mode],
        p=p
    )
    aa_spec = AASpec(target_success=0.99, max_iters=32)

    # Twirling configuration
    if args.no_twirl:
        twirling_spec = TwirlingSpec(enabled=False)
    else:
        twirling_spec = TwirlingSpec(enabled=True, mode="cyclic")

    # Build RunSpec
    spec = RunSpec(
        target=target_spec,
        noise=noise_spec,
        aa=aa_spec,
        twirling=twirling_spec,
        N=N,
        out_dir=Path(args.out),
        iterative_noise=args.iterative,
        purification_level=args.purification_level,
        verbose=args.verbose
    )

    # Validate before returning
    spec.validate()
    return spec
```

### Iterative Mode Parameters

When `--iterative` flag is enabled, parameter interpretation changes:

- **N parameter**: Controls number of iterations (log₂(N)) instead of total copies
- **purification_level**: Sets ℓ (SWAP rounds per iteration)
- **Total SWAP operations**: log₂(N) × ℓ

**Validation for Iterative Mode:**

```python
if args.iterative:
    # Ensure N is power of 2
    for N in n_list:
        if not (N > 0 and (N & (N - 1)) == 0):
            raise ValueError(f"N must be power of 2 for iterative mode, got {N}")

    # Validate purification level
    if args.purification_level < 1:
        raise ValueError("Purification level must be ≥ 1")
```

### Batch Execution and Error Handling

```python
def main() -> None:
    """Main execution function with comprehensive error handling."""

    # Parse arguments and setup logging
    args = parse_arguments()
    setup_logging(verbose=args.verbose)

    # Generate parameter grid
    param_combinations = generate_parameter_grid(args)
    total_runs = len(param_combinations)

    logger.info(f"Starting grid execution: {total_runs} total runs")
    logger.info(f"Output directory: {args.out}")

    # Track execution statistics
    completed = 0
    failed = 0
    start_time = time.time()

    # Main execution loop
    for idx, (M, N, p, noise_type) in enumerate(param_combinations):
        # Build configuration
        try:
            spec = build_run_spec(M, N, p, noise_type, args)
            tag = f"[{idx+1}/{total_runs}] {spec.synthesize_run_id()}"

            logger.info(f"Starting {tag}")
            t0 = time.time()

            # Execute simulation
            run_and_save(spec)

            # Record success
            dt = time.time() - t0
            completed += 1
            logger.info(f"✓ {tag} completed in {dt:.1f}s")

        except Exception as e:
            # Record failure and continue
            failed += 1
            logger.error(f"✗ {tag} failed: {e}", exc_info=True if args.verbose else False)

    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\nExecution complete:")
    logger.info(f"  Completed: {completed}/{total_runs} ({100*completed/total_runs:.1f}%)")
    logger.info(f"  Failed: {failed}/{total_runs} ({100*failed/total_runs:.1f}%)")
    logger.info(f"  Total time: {total_time:.1f}s")
```

The execution loop implements robust error handling and progress tracking:

1. **Parameter Validation**: Validate M values, noise types, and mode consistency
2. **RunSpec Generation**: Create RunSpec for each parameter combination
3. **Individual Execution**: Call `run_and_save()` with error catching
4. **Progress Logging**: Report completion statistics and timing

**Error Recovery Strategy:**

```python
try:
    run_and_save(spec)
    dt = time.time() - t0
    logger.info(f"✓ Completed in {dt:.1f}s")
except Exception as e:
    logger.error(f"✗ ERROR during {tag}: {e}", exc_info=True)
    # Continue with next run - don't terminate entire grid
```

This ensures that individual run failures don't terminate the entire grid sweep, allowing maximum data collection even when some parameter combinations encounter issues.

### Output File Management

```python
def run_and_save(spec: RunSpec) -> Tuple[Path, Path]:
    """Execute simulation and save results to CSV files."""

    # Execute simulation
    steps_df, finals_df = run_streaming(spec)

    # Ensure output directory exists
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    # Generate noise-specific file paths
    noise_suffix = spec.noise.noise_type.value
    steps_path = spec.out_dir / f"steps_circuit_{noise_suffix}.csv"
    finals_path = spec.out_dir / f"finals_circuit_{noise_suffix}.csv"

    # Incremental append to existing files
    for df, path in [(steps_df, steps_path), (finals_df, finals_path)]:
        if path.exists():
            # Load existing data and concatenate
            existing_df = pd.read_csv(path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            # Create new file
            combined_df = df

        # Save atomically
        combined_df.to_csv(path, index=False)

    return steps_path, finals_path
```

**Output File Naming Convention:**

- `steps_circuit_depolarizing.csv`: Step-by-step data for depolarizing noise
- `finals_circuit_depolarizing.csv`: Summary data for depolarizing noise
- `steps_circuit_dephase_z.csv`: Step-by-step data for Z-dephasing noise
- `finals_circuit_dephase_z.csv`: Summary data for Z-dephasing noise

**Critical Implementation Details:**

1. **Atomic Write**: Save both CSV files atomically to prevent corruption
2. **Incremental Append**: Concatenate with existing data if files exist
3. **Directory Creation**: Ensure output directory exists before writing
4. **Error Handling**: Graceful handling of file I/O errors

---

## System Integration & Data Flow

This section describes how all components integrate during a complete simulation run, tracing the data flow from command-line invocation through final CSV output.

### Complete Execution Flow

#### Phase 1: Initialization and Configuration

1. **CLI Parsing**: `main_grid_run.py` processes command-line arguments
2. **Parameter Grid Generation**: Construct Cartesian product of M, N, p, noise values
3. **RunSpec Creation**: For each parameter combination, create RunSpec with:
   - `TargetSpec(M, kind=hadamard, seed)`
   - `NoiseSpec(noise_type, mode=iid_p, p)`
   - `AASpec(target_success=0.99, max_iters=32)`
   - `TwirlingSpec(enabled=True, mode=cyclic)`
4. **Validation**: Call `spec.validate()` to check parameter consistency

#### Phase 2: Target State Preparation

1. **State Factory Invocation**: Call `build_target(spec.target)`
2. **Circuit Construction**: Generate preparation circuit based on `spec.target.kind`
3. **Reference Computation**: Compute exact Statevector |ψ⟩ for metrics

**Example for Hadamard states:**

- M=1: Circuit with single H gate, |ψ⟩ = |+⟩
- M=5: Circuit with 5 H gates, |ψ⟩ = |+⟩^⊗5

#### Phase 3A: Regular Streaming Mode

For `spec.iterative_noise = False`:

1. **Copy Generation**: Generate N noisy copies using `noise_engine.py`
   - If iid_p + no twirling: Generate single cached circuit, reuse N times
   - If twirling enabled: Generate N independent circuits with different twirl seeds
   - If exact_k mode: Ensure identical error patterns at level 0 merges

2. **Streaming Tree Execution**: Process copies through binary tree
   - Maintain slots dictionary mapping level → waiting density matrix
   - For each incoming copy: find empty level or merge with existing copy
   - Call `purify_two_from_density()` for each merge operation

#### Phase 3B: Iterative Purification Mode

For `spec.iterative_noise = True`:

1. **Iteration Loop**: Repeat log₂(N) times:
   - **Frame rotation**: Apply global Clifford C_t = {I,H,SH}[t mod 3]
   - **Noise application**: Call `apply_noise_to_density_matrix(current_state)`
   - **Frame restoration**: Apply C_t†
   - **Copy cloning**: Create 2^ℓ identical copies of noisy state
   - **Clean purification**: Apply streaming tree with ℓ levels

#### Phase 4: SWAP Test Purification

Each call to `purify_two_from_density()` executes:

1. **Joint State Construction**: Build |0⟩⟨0|\_anc ⊗ ρ_A ⊗ ρ_B
2. **SWAP Unitary Application**: Apply U_swap = H_anc × CSWAP × H_anc
3. **Success Probability**: Compute P₀ = Tr(Π₀ ρ) where Π₀ = |0⟩⟨0|\_anc ⊗ I ⊗ I
4. **Amplitude Amplification**: Calculate optimal Grover iterations k
5. **State Projection**: Project ancilla to |0⟩ and renormalize
6. **Register Extraction**: Trace out ancilla and register B, return register A

#### Phase 5: Metrics and Output

1. **State Characterization**: For each intermediate and final state:
   - Fidelity: F = ⟨ψ|ρ|ψ⟩
   - Trace distance: ε_L = ½||ρ - |ψ⟩⟨ψ|||₁
   - Purity: Tr(ρ²)
   - Bloch magnitude: |r⃗| (for M=1 only)

2. **DataFrame Construction**: Populate steps and finals DataFrames
3. **CSV Output**: Save to noise-specific files with incremental append

### Data Dependencies and Interfaces

The system maintains clean interfaces between components:

**configs.py ↔ All Modules:**

- Provides type-safe parameter containers
- Validates parameter consistency across modules
- Manages δ ↔ p conversions for theoretical consistency

**state_factory.py → streaming_runner.py:**

- Returns `(QuantumCircuit, Statevector)` tuple
- Circuit used for noise application, Statevector for metrics

**noise_engine.py ↔ streaming_runner.py:**

- Dual interface: circuit-based and density matrix-based noise
- Regular mode uses `build_noisy_copy()` for circuits
- Iterative mode uses `apply_noise_to_density_matrix()` for direct application

**amplified_swap.py ↔ streaming_runner.py:**

- Receives two DensityMatrix objects (inputs to purification)
- Returns `(purified_state, metrics_dict)` tuple
- Metrics include P_success and grover_iters for logging

### Critical Data Flow Dependencies

1. **Parameter Consistency**: All modules must use same δ/p conversion conventions
2. **State Normalization**: Density matrices must remain properly normalized through all operations
3. **Qubit Ordering**: Consistent qubit indexing across circuit construction and matrix operations
4. **Random Seeding**: Reproducible results require careful seed management across modules
5. **Memory Management**: Density matrix operations scale as 4^M, requiring careful resource planning

---

## CSV Output Specification

The simulation generates comprehensive CSV datasets with two distinct schemas optimized for different analysis purposes. All files use standard CSV format with headers and are designed for direct import into pandas, R, or other data analysis tools.

### Steps CSV Schema

The steps CSV contains one row per purification operation (merge or iteration), providing detailed trajectory information.

| Column               | Type       | Description                                                                                |
| -------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| `run_id`             | string     | Unique identifier synthesized from parameters (e.g., 'M1_N64_depolarizing_iid_p_p0.50000') |
| `merge_num`          | int        | Sequential merge operation number (regular mode) or iteration number (iterative mode)      |
| `M`                  | int        | Number of qubits in target system                                                          |
| `depth`              | int        | Purification depth: tree level (regular) or cumulative SWAP count (iterative)              |
| `copies_used`        | int        | Total number of noisy copies consumed to reach this point                                  |
| `N_so_far`           | int        | Alternative copy count measure for progress tracking                                       |
| `noise`              | string     | Noise type: 'depolarizing', 'dephase_z', or 'dephase_x'                                    |
| `mode`               | string     | Noise application mode: 'iid_p' or 'exact_k'                                               |
| `p`                  | float      | Channel probability parameter (primary error rate parameter)                               |
| `p_channel`          | float      | Same as p (maintained for backward compatibility)                                          |
| `P_success`          | float      | SWAP test success probability P[ancilla=0] for this merge                                  |
| `grover_iters`       | int        | Optimal Grover iterations for amplitude amplification (emulated)                           |
| `twirling_applied`   | bool       | Whether Clifford twirling was active for this run                                          |
| `fidelity`           | float      | Fidelity F = ⟨ψ\|ρ\|ψ⟩ to target pure state                                                |
| `eps_L`              | float      | Trace distance ε_L = ½\|\|ρ - \|ψ⟩⟨ψ\|\|\|₁ (error rate measure)                           |
| `purity`             | float      | Purity Tr(ρ²) quantifying degree of mixedness (1.0 = pure state)                           |
| `bloch_r`            | float/null | Bloch vector magnitude \|r⃗\| (M=1 only, null otherwise)                                   |
| `iteration`          | int/null   | Iteration round number (iterative mode only)                                               |
| `purification_level` | int/null   | ℓ parameter: SWAP rounds per iteration (iterative mode only)                               |

### Finals CSV Schema

The finals CSV contains one row per complete simulation run, providing summary statistics and comparative metrics.

| Column                  | Type       | Description                                                                        |
| ----------------------- | ---------- | ---------------------------------------------------------------------------------- |
| `run_id`                | string     | Unique identifier matching steps CSV                                               |
| `M`                     | int        | Number of qubits in target system                                                  |
| `N`                     | int        | Total copies used (regular) or iteration parameter (iterative)                     |
| `noise`                 | string     | Noise type identifier                                                              |
| `mode`                  | string     | Noise application mode                                                             |
| `p`                     | float      | Channel probability parameter                                                      |
| `p_channel`             | float      | Same as p (backward compatibility)                                                 |
| `twirling_applied`      | bool       | Whether Clifford twirling was enabled                                              |
| `fidelity_init`         | float      | Initial fidelity of noisy states before purification                               |
| `fidelity_final`        | float      | Final fidelity after complete purification protocol                                |
| `eps_L_init`            | float      | Initial trace distance (error rate) before purification                            |
| `eps_L_final`           | float      | Final trace distance (error rate) after purification                               |
| `error_reduction_ratio` | float      | Ratio eps_L_final / eps_L_init (< 1 indicates successful error reduction)          |
| `purity_init`           | float      | Initial purity of noisy states                                                     |
| `purity_final`          | float      | Final purity after purification                                                    |
| `bloch_r_init`          | float/null | Initial Bloch magnitude (M=1 only)                                                 |
| `bloch_r_final`         | float/null | Final Bloch magnitude (M=1 only)                                                   |
| `max_depth`             | int        | Maximum purification depth achieved: log₂(N) (regular) or iterations×ℓ (iterative) |
| `total_merges`          | int        | Total number of SWAP operations performed                                          |
| `iterations`            | int/null   | Total number of iterations performed (iterative mode only)                         |
| `purification_level`    | int/null   | ℓ parameter (iterative mode only)                                                  |

### File Organization and Naming

CSV files are organized by noise type to facilitate efficient loading and analysis:

- **`steps_circuit_depolarizing.csv`**: All step-by-step data for depolarizing noise runs
- **`finals_circuit_depolarizing.csv`**: All summary data for depolarizing noise runs
- **`steps_circuit_dephase_z.csv`**: All step-by-step data for Z-dephasing noise runs
- **`finals_circuit_dephase_z.csv`**: All summary data for Z-dephasing noise runs

Each file grows incrementally as new simulation runs complete, allowing for interrupted grid sweeps and partial analysis.

### Data Analysis Recommendations

The CSV structure facilitates various analysis workflows:

**Threshold Analysis:**

- Use finals CSV with filters on M and N
- Plot error_reduction_ratio vs p to identify threshold regions
- Compare different M values to study scaling

**Purification Trajectory Analysis:**

- Use steps CSV grouped by run_id
- Plot fidelity vs depth for specific parameter combinations
- Analyze convergence behavior and purification efficiency

**Iterative Protocol Analysis:**

- Filter steps CSV for iterative_noise mode
- Group by purification_level to compare different ℓ values
- Plot fidelity vs iteration for fixed ℓ and varying p

**Comparative Studies:**

- Compare regular vs iterative modes with equivalent total SWAP counts
- Analyze twirling effectiveness by comparing twirling_applied groups
- Study noise type dependencies by loading different CSV files

### Example Analysis Code

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
steps_depol = pd.read_csv('data/steps_circuit_depolarizing.csv')
finals_depol = pd.read_csv('data/finals_circuit_depolarizing.csv')

# Threshold analysis
plt.figure(figsize=(10, 6))
for M in [1, 5]:
    data = finals_depol[(finals_depol['M'] == M) & (finals_depol['N'] == 64)]
    plt.loglog(data['p'], data['error_reduction_ratio'],
               marker='o', label=f'M={M}')

plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No improvement')
plt.xlabel('Physical Error Rate p')
plt.ylabel('Error Reduction Ratio')
plt.legend()
plt.title('PEC Threshold Analysis')
plt.show()

# Iterative trajectory analysis
iterative_data = steps_depol[
    (steps_depol['iteration'].notna()) &
    (steps_depol['M'] == 1) &
    (steps_depol['p'] == 0.5)
]

plt.figure(figsize=(10, 6))
for ell in iterative_data['purification_level'].unique():
    data = iterative_data[iterative_data['purification_level'] == ell]
    plt.plot(data['iteration'], data['fidelity'],
             marker='s', label=f'ℓ={ell}')

plt.xlabel('Iteration Round')
plt.ylabel('Fidelity')
plt.legend()
plt.title('Iterative Purification Trajectories (M=1, p=0.5)')
plt.show()
```

This comprehensive dataset enables detailed analysis of quantum error correction performance across the full parameter space, supporting both theoretical validation and practical implementation studies.

---

## Complete Execution Flow

### Summary of End-to-End Execution

When you run the main grid execution, here's the complete flow:

1. **Command Parsing** → Parse CLI arguments, set up logging
2. **Grid Generation** → Create all (M, N, p, noise_type) combinations
3. **Parameter Loop** → For each combination:
   - Build RunSpec with all configuration objects
   - Validate parameters for consistency
   - Generate target state using state_factory
   - Choose execution mode (regular vs iterative)
   - Apply noise and purification protocol
   - Compute comprehensive metrics
   - Save results to appropriate CSV files
4. **Error Handling** → Continue on individual failures, report statistics
5. **Output** → Noise-specific CSV files with complete datasets

### Key Design Principles

- **Modularity**: Clear separation between configuration, state generation, noise application, and purification
- **Theoretical Rigor**: Explicit mathematical implementations avoid hidden assumptions
- **Flexibility**: Support for multiple noise models, state types, and protocol variants
- **Robustness**: Comprehensive validation and error handling throughout
- **Performance**: Efficient density matrix operations and memory management
- **Reproducibility**: Deterministic seeding and parameter tracking

### Critical Implementation Notes

1. **Memory Scaling**: Density matrices scale as 4^M, limiting M ≤ 6 for practical simulations
2. **SWAP Test Theory**: Requires identical input states for theoretical validity
3. **Clifford Twirling**: Essential for dephasing noise mitigation, implemented per manuscript Eq. 54
4. **Amplitude Amplification**: Emulated rather than simulated for computational efficiency
5. **Parameter Consistency**: δ/p conversions maintained across all modules
6. **Iterative Protocol**: Global Clifford cycling provides deterministic, reproducible results

---

This completes the comprehensive technical documentation of the SWAP-based quantum error correction simulation framework. Every function, parameter, data flow, and implementation detail has been covered to ensure complete understanding of the codebase functionality.
