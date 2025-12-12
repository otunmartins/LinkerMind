import pandas as pd
import numpy as np
import os

def analyze_current_data():
    """Analyze your current dataset structure"""
    print("=== Analyzing Current Dataset Structure ===\n")
    
    # List all CSV files
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print("Available CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Check key files
    key_files = [
        'positive_linkers_combined.csv',
        'linkermind_final_dataset_fixed.csv', 
        'negative_decoys.csv'
    ]
    
    for file in key_files:
        if os.path.exists(file):
            print(f"\n=== Analyzing {file} ===")
            df = pd.read_csv(file)
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            print("First few rows:")
            print(df.head(2))
            print("\n" + "="*50)
        else:
            print(f"\n{file} not found")

if __name__ == "__main__":
    analyze_current_data()
