import os
import json
from trinetra.layer2_model_security.cli import main
import subprocess

def run_demo():
    print("=== TRI-NETRA Layer 2 Demo ===")
    
    # 1. Generate models
    print("\n--- Generating Models ---")
    subprocess.run(["python", "-m", "trinetra.layer2_model_security.cli", "attack", "--out", "outputs/layer2/demo_models"])
    
    # 2. Scan clean model
    print("\n--- Scanning Clean Model ---")
    subprocess.run(["python", "-m", "trinetra.layer2_model_security.cli", "scan", "--model", "outputs/layer2/demo_models/clean_model.pt"])
    
    # 3. Scan backdoored model
    print("\n--- Scanning Backdoored Model ---")
    subprocess.run(["python", "-m", "trinetra.layer2_model_security.cli", "scan", "--model", "outputs/layer2/demo_models/backdoored_model.pt"])
    
    # 4. Scan locked model
    print("\n--- Scanning Locked Model (Black-box Fallback) ---")
    subprocess.run(["python", "-m", "trinetra.layer2_model_security.cli", "scan", "--model", "outputs/layer2/demo_models/backdoored_model_locked.bin", "--mode", "auto"])
    
    # 5. Heal the backdoored model
    print("\n--- Healing Backdoored Model ---")
    subprocess.run(["python", "-m", "trinetra.layer2_model_security.cli", "heal", "--model", "outputs/layer2/demo_models/backdoored_model.pt", "--report", "outputs/layer2/layer2_report.json"])
    
    # 6. Rescan healed model
    print("\n--- Rescanning Healed Model ---")
    subprocess.run(["python", "-m", "trinetra.layer2_model_security.cli", "scan", "--model", "outputs/layer2/healed_model.pt"])
    
if __name__ == "__main__":
    run_demo()
