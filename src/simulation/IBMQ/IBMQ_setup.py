# """
# SETUP GUIDE: IBM Quantum SWAP-based PEC Experiments
# =================================================

# This guide provides step-by-step instructions for setting up and running your
# sequential SWAP-based purification experiments on IBM Quantum hardware.
# """

# # =============================================================================
# # 1. ENVIRONMENT SETUP
# # =============================================================================

# """
# First, set up your environment with the required packages:

# pip install qiskit qiskit-ibm-runtime qiskit-ibm-provider matplotlib pandas numpy

# You'll need:
# 1. An IBM Quantum account (free): https://quantum-computing.ibm.com/
# 2. Your API token from the account dashboard
# """

# # =============================================================================
# # 2. IBM QUANTUM ACCOUNT CONFIGURATION
# # =============================================================================

# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit_ibm_provider import IBMProvider

# # First-time setup (run once)
# def setup_ibm_account():
#     """
#     Set up IBM Quantum account credentials.
    
#     Get your token from: https://quantum-computing.ibm.com/account
#     """
#     # Replace 'YOUR_TOKEN_HERE' with your actual IBM Quantum token
#     token = 'YOUR_TOKEN_HERE'
    
#     # Save credentials
#     QiskitRuntimeService.save_account(token=token, overwrite=True)
    
#     print("IBM Quantum account configured successfully!")
    
#     # Test connection
#     service = QiskitRuntimeService()
#     backends = service.backends(simulator=False, operational=True)
#     print(f"Available backends: {[b.name for b in backends[:5]]}")

# # =============================================================================
# # 3. RECOMMENDED EXPERIMENTAL SEQUENCE
# # =============================================================================

# """
# Start with this progression to validate your implementation:

# PHASE 1: Simulator Validation (1-2 hours)
# - Test on 'ibmq_qasm_simulator' first
# - Verify circuit construction and data collection
# - Debug any issues with low overhead

# PHASE 2: Small-Scale Hardware (4-8 hours)
# - Start with M=1 (single qubit) experiments
# - Use a 5-qubit backend like 'ibm_nairobi' or 'ibm_oslo' 
# - Test different noise levels: p = [0.05, 0.1, 0.2]

# PHASE 3: Multi-Qubit Validation (8-16 hours)
# - Expand to M=2, then M=3 if results are promising
# - Focus on backends with good connectivity (ibm_brisbane, ibm_kyoto)
# - Compare depolarizing vs dephasing + twirling

# PHASE 4: Comprehensive Study (days-weeks)
# - Full parameter sweeps
# - Statistical significance studies
# - Hardware characterization correlation
# """

# # =============================================================================
# # 4. BACKEND SELECTION GUIDE
# # =============================================================================

# RECOMMENDED_BACKENDS = {
#     # For M=1 experiments (any backend works)
#     "single_qubit": [
#         "ibm_nairobi",    # 7 qubits, good performance
#         "ibm_oslo",       # 7 qubits, reliable
#         "ibm_perth",      # 7 qubits, stable
#     ],
    
#     # For M=2 experiments (need good 2-qubit connectivity)
#     "two_qubit": [
#         "ibm_brisbane",   # 127 qubits, excellent connectivity
#         "ibm_kyoto",      # 127 qubits, good performance
#         "ibm_sherbrooke", # 127 qubits, newer backend
#     ],
    
#     # For M=3 experiments (need 3+ connected qubits)
#     "three_qubit": [
#         "ibm_brisbane",   # Best choice for multi-qubit
#         "ibm_kyoto",      # Good alternative
#     ]
# }

# def get_backend_recommendations(M: int):
#     """Get recommended backends for M-qubit experiments."""
#     if M == 1:
#         return RECOMMENDED_BACKENDS["single_qubit"]
#     elif M == 2:
#         return RECOMMENDED_BACKENDS["two_qubit"] 
#     else:
#         return RECOMMENDED_BACKENDS["three_qubit"]

# # =============================================================================
# # 5. QUICK START EXAMPLE
# # =============================================================================

# def quick_start_example():
#     """
#     Minimal example to test your setup and run a single experiment.
#     """
#     from pathlib import Path
    
#     # Import your modules
#     from configs import TargetSpec, NoiseSpec, AASpec, RunSpec, StateKind, NoiseType, TwirlingSpec
#     from ibmq_pec_implementation import IBMQSpec, HardwarePECExperiment, save_hardware_results
    
#     print("🚀 Quick Start: Single Qubit SWAP-based PEC")
#     print("=" * 50)
    
#     # Configure a simple experiment
#     target = TargetSpec(M=1, kind=StateKind.hadamard, seed=42)
#     noise = NoiseSpec(noise_type=NoiseType.depolarizing, p=0.1)
#     aa = AASpec(target_success=0.9, max_iters=16)
#     twirling = TwirlingSpec(enabled=False)  # Disabled for depolarizing
    
#     run_spec = RunSpec(
#         target=target,
#         noise=noise, 
#         aa=aa,
#         twirling=twirling,
#         N=4,  # Small N for quick test
#         verbose=True
#     )
    
#     # Hardware configuration - START WITH SIMULATOR
#     ibmq_spec = IBMQSpec(
#         backend_name="ibmq_qasm_simulator",  # Safe choice for testing
#         use_simulator=True,
#         shots_per_circuit=1024,
#         characterize_noise=False  # Skip for simulator
#     )
    
#     try:
#         # Run experiment
#         print("⏳ Running experiment...")
#         experiment = HardwarePECExperiment(run_spec, ibmq_spec)
#         result = experiment.run_single_round(num_rounds=3)
        
#         # Save results
#         output_dir = Path("data/quick_start_test")
#         save_hardware_results(result, output_dir)
        
#         # Display results
#         print("✅ Experiment completed successfully!")
#         print(f"   Mean fidelity: {np.mean(result.fidelity_estimates):.3f}")
#         print(f"   Mean success rate: {np.mean(result.success_rates):.3f}")
#         print(f"   Results saved to: {output_dir}")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Experiment failed: {e}")
#         print("💡 Troubleshooting tips:")
#         print("   1. Check your IBM Quantum token")
#         print("   2. Verify internet connection")
#         print("   3. Try 'ibmq_qasm_simulator' backend first")
#         return False

# # =============================================================================
# # 6. EXPERIMENTAL BEST PRACTICES
# # =============================================================================

# """
# KEY RECOMMENDATIONS:

# 1. CIRCUIT DEPTH MANAGEMENT:
#    - Keep total circuit depth < 100 gates for reliable results
#    - SWAP test adds significant depth (2M CNOT gates + preparations)
#    - Consider using SWAP test approximations for M > 3

# 2. STATISTICAL CONSIDERATIONS:
#    - Use ≥ 1024 shots per circuit for decent statistics  
#    - Run each configuration 3-5 times to assess reproducibility
#    - Account for IBM Q calibration drift (results vary by day)

# 3. QUEUE MANAGEMENT:
#    - Submit jobs in batches to minimize queue waiting
#    - Use job.status() to monitor progress
#    - Consider overnight runs for comprehensive studies

# 4. DATA VALIDATION:
#    - Always compare early results with simulation
#    - Check for systematic biases (e.g., always getting 50/50 measurements)
#    - Monitor backend calibration data for correlation with your results

# 5. ERROR SOURCES TO CONSIDER:
#    - Gate errors (typically 0.1-1% for 1-qubit, 1-5% for 2-qubit)
#    - Readout errors (typically 1-5%)
#    - Coherence limits (T1 ~ 100μs, T2 ~ 100μs)
#    - Crosstalk between qubits
#    - Calibration drift during long experiments
# """

# # =============================================================================
# # 7. TROUBLESHOOTING COMMON ISSUES
# # =============================================================================

# TROUBLESHOOTING_GUIDE = {
#     "Authentication Error": {
#         "symptoms": "Cannot connect to IBM Quantum",
#         "solutions": [
#             "Verify your API token is correct",
#             "Check internet connection",
#             "Try: QiskitRuntimeService.delete_account() then re-save",
#             "Ensure you have access to the requested backend"
#         ]
#     },
    
#     "Job Timeout": {
#         "symptoms": "Jobs hang in queue for hours",
#         "solutions": [
#             "Choose less busy backends (check queue lengths)",
#             "Reduce shots_per_circuit or max_circuits_per_job",
#             "Submit smaller batches",
#             "Try during off-peak hours (US nighttime)"
#         ]
#     },
    
#     "Poor Fidelity Results": {
#         "symptoms": "Fidelities much lower than simulation",
#         "solutions": [
#             "Check backend calibration data",
#             "Reduce circuit depth (use fewer purification rounds)",
#             "Verify noise model assumptions",
#             "Compare with process tomography benchmarks"
#         ]
#     },
    
#     "Inconsistent Results": {
#         "symptoms": "Large variance between repeated runs",
#         "solutions": [
#             "Increase shots_per_circuit",
#             "Check for backend maintenance periods",
#             "Monitor T1/T2 coherence times",
#             "Use error mitigation techniques"
#         ]
#     }
# }

# def print_troubleshooting_guide():
#     """Print the troubleshooting guide."""
#     print("🔧 TROUBLESHOOTING GUIDE")
#     print("=" * 50)
    
#     for issue, info in TROUBLESHOOTING_GUIDE.items():
#         print(f"\n{issue}:")
#         print(f"  Symptoms: {info['symptoms']}")
#         print(f"  Solutions:")
#         for solution in info['solutions']:
#             print(f"    • {solution}")

# # =============================================================================
# # 8. RUNNING THE FULL EXPERIMENTAL SUITE
# # =============================================================================

# if __name__ == "__main__":
#     import sys
    
#     print("IBM Quantum SWAP-based PEC Setup Guide")
#     print("=" * 50)
    
#     # Check command line arguments for different modes
#     if len(sys.argv) > 1:
#         if sys.argv[1] == "setup":
#             setup_ibm_account()
#         elif sys.argv[1] == "test":
#             quick_start_example()
#         elif sys.argv[1] == "troubleshoot":
#             print_troubleshooting_guide()
#         elif sys.argv[1] == "backends":
#             for M in [1, 2, 3]:
#                 backends = get_backend_recommendations(M)
#                 print(f"M={M}: {backends}")
#     else:
#         print("\nAvailable commands:")
#         print("  python setup_guide.py setup       - Configure IBM Quantum account")
#         print("  python setup_guide.py test        - Run quick test experiment") 
#         print("  python setup_guide.py backends    - Show recommended backends")
#         print("  python setup_guide.py troubleshoot - Show troubleshooting guide")
#         print("\nFor full experiments, run: python run_hardware_experiments.py")
        
        
# =========================

"""
IBM Quantum account setup utility.

Usage:
  python IBMQ_setup.py save <API_TOKEN> [--instance ORG/PROJECT/HUB] [--channel CHANNEL]
  python IBMQ_setup.py test [--backend BACKEND_NAME]
  python IBMQ_setup.py delete

Examples:
  python IBMQ_setup.py save sk-XXXX... --instance ibm-q/open/main
  python IBMQ_setup.py test --backend ibmq_qasm_simulator
"""

from __future__ import annotations

import argparse
from typing import Optional

from qiskit_ibm_runtime import QiskitRuntimeService

VALID_CHANNELS = ("ibm_quantum_platform", "ibm_cloud")


def save_account(token: str, instance: Optional[str] = None, channel: Optional[str] = None) -> None:
    """Save IBM Quantum account credentials for the runtime service.

    Parameters
    ----------
    token : str
        Your API token from https://quantum-computing.ibm.com/account
        (or your IBM Cloud API key if using the Cloud channel).
    instance : Optional[str]
        Optional hub/group/project instance string, e.g. "ibm-q/open/main"
        for the IBM Quantum Platform. For IBM Cloud, this should be the CRN.
    channel : Optional[str]
        One of {"ibm_quantum_platform","ibm_cloud"}. If omitted, defaults
        to "ibm_quantum_platform".
    """
    ch = channel or "ibm_quantum_platform"
    if ch not in VALID_CHANNELS:
        raise SystemExit(f"Invalid channel '{ch}'. Expected one of {VALID_CHANNELS}.")

    kwargs = {"token": token, "channel": ch}
    if instance:
        kwargs["instance"] = instance

    QiskitRuntimeService.save_account(**kwargs, overwrite=True)
    print(f"✅ Saved IBM account for qiskit-ibm-runtime (channel={ch}"
          f"{', instance='+instance if instance else ''}).")


def delete_account() -> None:
    """Delete any saved IBM Quantum account credentials."""
    try:
        QiskitRuntimeService.delete_account()
        print("🗑️  Deleted saved qiskit-ibm-runtime account credentials.")
    except Exception as e:
        print(f"⚠️  Could not delete credentials (maybe none saved?): {e}")


def test_connection(backend_name: Optional[str] = None) -> None:
    """Test connectivity and list backends. If backend_name is provided,
    attempt to instantiate it to confirm access.
    """
    svc = QiskitRuntimeService()
    # Prefer operational backends to reduce noise
    try:
        bks = svc.backends(operational=True)
    except TypeError:
        # Older versions may not support 'operational' kwarg
        bks = svc.backends()

    print(f"✅ Runtime service initialized. Found {len(bks)} backends.")
    sample = ", ".join(b.name for b in bks[:8])
    if sample:
        print("Some operational backends:", sample)

    if backend_name:
        try:
            backend = svc.backend(backend_name)
            # A lightweight poke: access a couple of attributes
            _ = backend.name
            _ = getattr(backend, "num_qubits", "n/a")
            print(f"✅ Connected to backend: {backend.name}")
        except Exception as e:
            print(f"❌ Failed to access backend '{backend_name}': {e}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IBM Quantum account setup utility")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_save = sub.add_parser("save", help="Save API token")
    s_save.add_argument("token", type=str, help="API token (or IBM Cloud API key)")
    s_save.add_argument("--instance", type=str, default=None,
                        help="For Platform: 'ibm-q/open/main'. For Cloud: CRN string.")
    s_save.add_argument("--channel", choices=VALID_CHANNELS, default=None,
                        help="Override channel (default: ibm_quantum_platform)")

    s_test = sub.add_parser("test", help="Test connection")
    s_test.add_argument("--backend", type=str, default=None,
                        help="Optional backend name to verify access")

    sub.add_parser("delete", help="Delete saved credentials")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "save":
        save_account(args.token, args.instance, args.channel)
        # Do a quick sanity check right away
        test_connection()
    elif args.cmd == "test":
        test_connection(args.backend)
    elif args.cmd == "delete":
        delete_account()


if __name__ == "__main__":
    main()