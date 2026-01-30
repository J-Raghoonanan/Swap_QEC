#!/usr/bin/env python3
"""
Check IBM backend status to diagnose hanging issues.
This helps identify if the backend is available before running experiments.
"""

import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_backend_status(backend_name: str = "ibm_torino") -> Dict:
    """Check the status and queue length of an IBM backend."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        logger.info(f"🔍 Checking status of {backend_name}...")
        
        # Connect to IBM service
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        
        # Get backend status
        status = backend.status()
        
        print(f"\n📊 Backend Status: {backend_name}")
        print(f"{'='*50}")
        print(f"🟢 Operational: {status.operational}")
        print(f"📝 Status Message: {status.status_msg}")
        print(f"⏳ Pending Jobs: {status.pending_jobs}")
        print(f"🔧 Processor Type: {backend.configuration().processor_type}")
        print(f"💾 Max Shots: {backend.configuration().max_shots}")
        print(f"🎯 Qubits: {backend.configuration().n_qubits}")
        
        # Estimate wait time based on pending jobs
        if status.pending_jobs == 0:
            wait_estimate = "Immediate"
        elif status.pending_jobs < 5:
            wait_estimate = "< 30 minutes"
        elif status.pending_jobs < 20:
            wait_estimate = "30-60 minutes"
        else:
            wait_estimate = "> 1 hour"
        
        print(f"⏰ Estimated Wait: {wait_estimate}")
        
        # Recommendation
        if not status.operational:
            print(f"❌ RECOMMENDATION: Backend is down - try different backend")
            return {"status": "down", "pending": status.pending_jobs}
        elif status.pending_jobs > 50:
            print(f"⚠️ RECOMMENDATION: Very busy - consider smaller backend or simulator")
            return {"status": "busy", "pending": status.pending_jobs}
        elif status.pending_jobs > 10:
            print(f"🟡 RECOMMENDATION: Moderately busy - expect delays")
            return {"status": "moderate", "pending": status.pending_jobs}
        else:
            print(f"✅ RECOMMENDATION: Good time to submit jobs")
            return {"status": "good", "pending": status.pending_jobs}
            
    except ImportError:
        print(f"❌ IBM Runtime not available - cannot check backend status")
        return {"status": "unknown", "error": "IBM Runtime not available"}
    except Exception as e:
        print(f"❌ Error checking backend: {e}")
        return {"status": "error", "error": str(e)}

def suggest_alternatives() -> List[str]:
    """Suggest alternative backends that might be less busy."""
    alternatives = [
        "ibm_brisbane",     # 127 qubits
        "ibm_kyoto",        # 127 qubits  
        "ibm_osaka",        # 127 qubits
        "ibm_torino",       # 133 qubits (current)
    ]
    
    print(f"\n🔄 Alternative Backends to Try:")
    print(f"{'='*40}")
    for backend in alternatives:
        print(f"  - {backend}")
    
    return alternatives

def check_multiple_backends():
    """Check status of multiple backends to find the best one."""
    backends_to_check = ["ibm_torino", "ibm_brisbane", "ibm_kyoto", "ibm_osaka"]
    
    print(f"🔍 Checking Multiple Backend Status")
    print(f"{'='*60}")
    
    best_backend = None
    min_pending = float('inf')
    
    for backend_name in backends_to_check:
        try:
            status_info = check_backend_status(backend_name)
            pending = status_info.get('pending', float('inf'))
            
            if status_info['status'] == 'good' and pending < min_pending:
                min_pending = pending
                best_backend = backend_name
                
        except Exception as e:
            print(f"⚠️ Could not check {backend_name}: {e}")
        print("")  # Add spacing
    
    if best_backend:
        print(f"🏆 RECOMMENDATION: Use {best_backend} (only {min_pending} pending jobs)")
    else:
        print(f"⚠️ All backends appear busy - consider using simulator")

if __name__ == "__main__":
    print("🏥 IBM Quantum Backend Health Check")
    print("="*50)
    print("This tool diagnoses backend availability issues")
    print("that can cause job submission to hang.\n")
    
    # Check primary backend
    status_info = check_backend_status("ibm_torino")
    print()
    
    # Check alternatives if primary is busy
    if status_info.get('status') in ['busy', 'down', 'moderate']:
        check_multiple_backends()
    
    print(f"\n💡 If all backends are busy:")
    print(f"   - Use --quick mode for faster experiments") 
    print(f"   - Try during off-peak hours (US night time)")
    print(f"   - Use simulator for testing: python simulator_fallback_test.py")
    print(f"   - Reduce shots per experiment")