"""
Debug script to diagnose the SWAP test issue.

This manually traces through a single M=1 SWAP test to see where the discrepancy occurs.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, Operator, partial_trace

# Build a simple noisy state: δ=0.5 depolarizing on |+⟩
delta = 0.5
psi = Statevector.from_label('+')  # (|0⟩+|1⟩)/√2

# Depolarizing: ρ = (1-δ)|ψ⟩⟨ψ| + δ(I/2)
pure_dm = DensityMatrix(psi)
mixed_dm = DensityMatrix(np.eye(2)/2)

rho = (1-delta) * pure_dm + delta * mixed_dm

print("="*70)
print("MANUAL SWAP TEST TRACE")
print("="*70)

print(f"\n1. Initial state ρ:")
print(f"   δ = {delta}")
print(f"   ρ matrix:")
print(rho.data)

# Compute purity
purity = np.real(np.trace(rho.data @ rho.data))
print(f"   Tr(ρ²) = {purity:.6f}")

# Compute Bloch vector
from qiskit.quantum_info import Pauli
rx = np.real(np.trace(rho.data @ Pauli('X').to_matrix()))
ry = np.real(np.trace(rho.data @ Pauli('Y').to_matrix()))
rz = np.real(np.trace(rho.data @ Pauli('Z').to_matrix()))
r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
print(f"   Bloch vector: ({rx:.4f}, {ry:.4f}, {rz:.4f})")
print(f"   |r⃗| = {r_mag:.6f}")

# Theory prediction
r_out_theory = 4*r_mag / (3 + r_mag**2)
print(f"\n   Theory: |r⃗_out| = 4|r⃗|/(3+|r⃗|²) = {r_out_theory:.6f}")

# Expected P_success
P_theory = (1 + purity) / 2
print(f"   Theory: P_success = (1+Tr(ρ²))/2 = {P_theory:.6f}")

print(f"\n2. Joint state: |0⟩⟨0| ⊗ ρ ⊗ ρ")
# CRITICAL: Qiskit uses REVERSE ordering! 
# A.tensor(B) creates B ⊗ A (little-endian)
# We want |0⟩ ⊗ ρ_A ⊗ ρ_B in circuit notation (anc is qubit 0)
# So we must build it as: rho_B.tensor(rho_A).tensor(|0⟩)
# OR equivalently: use from_label with explicit basis ordering

# Let's use explicit construction to be sure of the ordering
# We want the matrix in the order |anc, A, B⟩
import itertools
basis_order = list(itertools.product([0,1], repeat=3))  # [(anc, A, B), ...]

# Build the 8x8 matrix explicitly
dim = 8
rho_joint_manual = np.zeros((dim, dim), dtype=complex)

for i, (anc_i, a_i, b_i) in enumerate(basis_order):
    for j, (anc_j, a_j, b_j) in enumerate(basis_order):
        # |0⟩⟨0| ⊗ ρ ⊗ ρ
        if anc_i == 0 and anc_j == 0:
            rho_joint_manual[i, j] = rho.data[a_i, a_j] * rho.data[b_i, b_j]

rho_joint = DensityMatrix(rho_joint_manual)
print(f"   Dimension: {rho_joint.dim} (should be 2³=8)")
print(f"   Built with EXPLICIT ordering: |anc=0, A, B⟩")

print(f"\n3. Build SWAP test unitary EXPLICITLY (bypass Qiskit circuit)")
# Build the unitary matrix for H ⊗ I ⊗ I
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I = np.eye(2, dtype=complex)
H_anc = np.kron(np.kron(H, I), I)  # H on qubit 0 (anc), I on qubits 1,2

# Build controlled-SWAP (Fredkin) gate
# Control = qubit 0, targets = qubits 1, 2
# When control=1, swap targets; when control=0, do nothing
# In computational basis |anc, A, B⟩:
# |0,a,b⟩ → |0,a,b⟩ (no swap)
# |1,a,b⟩ → |1,b,a⟩ (swap A and B)

CSWAP = np.eye(8, dtype=complex)
# Basis ordering: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩
#                 0      1      2      3      4      5      6      7
# Map when anc=1:
# |100⟩ (idx 4: anc=1,A=0,B=0) → |100⟩ (no change, both A and B are 0)
# |101⟩ (idx 5: anc=1,A=0,B=1) → |110⟩ (idx 6: swap A=0,B=1 → A=1,B=0)
# |110⟩ (idx 6: anc=1,A=1,B=0) → |101⟩ (idx 5: swap A=1,B=0 → A=0,B=1)
# |111⟩ (idx 7: anc=1,A=1,B=1) → |111⟩ (no change, both A and B are 1)

CSWAP[5, 5] = 0
CSWAP[6, 6] = 0
CSWAP[5, 6] = 1
CSWAP[6, 5] = 1

# Full SWAP test unitary: H × CSWAP × H
U_swap = H_anc @ CSWAP @ H_anc

print(f"   Built explicit SWAP test unitary (8×8 matrix)")
print(f"   U_swap shape: {U_swap.shape}")

# Apply to joint state
rho_after_swap = DensityMatrix(U_swap @ rho_joint.data @ U_swap.conj().T)
print(f"   Applied unitary evolution")

print(f"\n4. Compute P(ancilla=0) using CORRECT method: Tr(Π₊ ρ)")
# Build projector Π₊ = |0⟩⟨0| on ancilla, I on A and B
P0_anc = np.array([[1, 0], [0, 0]], dtype=complex)
I_A = np.eye(2, dtype=complex)
I_B = np.eye(2, dtype=complex)

# Full projector using Kronecker product (order: anc ⊗ A ⊗ B)
Pi_full = np.kron(np.kron(P0_anc, I_A), I_B)

# Compute Tr(Π ρ)
projected_for_prob = Pi_full @ rho_after_swap.data @ Pi_full.conj().T
p0_computed = float(np.real(np.trace(projected_for_prob)))

print(f"   P(ancilla=0) = Tr(Π₊ ρ) = {p0_computed:.6f}")
print(f"   Expected: {P_theory:.6f}")
print(f"   Match: {abs(p0_computed - P_theory) < 0.01}")

print(f"\n4b. [WRONG METHOD] Trace out A and B first (for comparison)")
anc_dm = partial_trace(rho_after_swap, qargs=[1, 2])
p0_wrong = float(np.real(anc_dm.data[0, 0]))
print(f"   P(ancilla=0) via marginal = {p0_wrong:.6f}")
print(f"   This is WRONG for controlled gates!")

print(f"\n5. Project ancilla onto |0⟩")
# Projector: |0⟩⟨0| on ancilla, I on A and B
P0_anc = np.array([[1, 0], [0, 0]], dtype=complex)
I_A = np.eye(2, dtype=complex)
I_B = np.eye(2, dtype=complex)

# Build full projector using Kronecker product
# Order: anc ⊗ A ⊗ B
Pi_full = np.kron(np.kron(P0_anc, I_A), I_B)

# Apply projection
rho_proj_data = Pi_full @ rho_after_swap.data @ Pi_full.conj().T

# Renormalize
p0_proj = np.real(np.trace(rho_proj_data))
print(f"   Trace of Π ρ Π† = {p0_proj:.6f}")
print(f"   (Should match P(ancilla=0) = {p0_computed:.6f})")

rho_proj = DensityMatrix(rho_proj_data / p0_proj)

print(f"\n6. Trace out register B, then ancilla, to get purified A")
# Step 1: Trace out register B (qubit 2)
rho_anc_A = partial_trace(rho_proj, qargs=[2])
print(f"   After tracing out B, dimension = {rho_anc_A.dim} (should be 4 for 2 qubits)")

# Step 2: Trace out ancilla (qubit 0)  
rho_A_out = partial_trace(rho_anc_A, qargs=[0])
print(f"   After tracing out ancilla, dimension = {rho_A_out.dim} (should be 2)")

print(f"   Output density matrix ρ_out:")
print(rho_A_out.data)

# Compute Bloch vector of output
rx_out = np.real(np.trace(rho_A_out.data @ Pauli('X').to_matrix()))
ry_out = np.real(np.trace(rho_A_out.data @ Pauli('Y').to_matrix()))
rz_out = np.real(np.trace(rho_A_out.data @ Pauli('Z').to_matrix()))
r_mag_out = np.sqrt(rx_out**2 + ry_out**2 + rz_out**2)

print(f"   Bloch vector: ({rx_out:.4f}, {ry_out:.4f}, {rz_out:.4f})")
print(f"   |r⃗_out| = {r_mag_out:.6f}")
print(f"   Expected: {r_out_theory:.6f}")
print(f"   Match: {abs(r_mag_out - r_out_theory) < 0.01}")

print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Initial |r⃗| = {r_mag:.6f}")
print(f"Theory  |r⃗_out| = {r_out_theory:.6f}")
print(f"Actual  |r⃗_out| = {r_mag_out:.6f}")
print(f"")
print(f"P_success theory = {P_theory:.6f}")
print(f"P_success actual = {p0_computed:.6f}")
print("="*70)