"""
Reproduction script for LinkerMind key results.
Run this to reproduce the main findings from the manuscript.
"""
import subprocess
import sys

def run_script(script_path):
    """Run a Python script and handle errors"""
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success: {script_path}")
            return True
        else:
            print(f"✗ Failed: {script_path}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running {script_path}: {e}")
        return False

def main():
    print("Reproducing LinkerMind key results...")
    
    steps = [
        ("Data preparation", "src/data/prepare_data.py"),
        ("Model training", "src/models/train_fusion_clean.py"),
        ("Generative design", "src/generative/calibrated_generative_pipeline.py"),
        ("Evaluation", "src/evaluation/clinical_validation_fixed.py"),
    ]
    
    for step_name, script_path in steps:
        print(f"\n--- {step_name} ---")
        if not run_script(script_path):
            print(f"Stopping reproduction due to failure in {step_name}")
            return
    
    print("\n🎉 Reproduction complete! Check results/ folder for outputs.")

if __name__ == "__main__":
    main()
