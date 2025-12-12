"""
Data preparation script for LinkerMind.
Loads the curated dataset and prepares features for model training.
"""
import pandas as pd
import pickle
import os

def load_dataset():
    """Load the main LinkerMind dataset"""
    dataset_path = "data/processed/linkermind_final_dataset_fixed.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} molecules")
        return df
    else:
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

def load_features():
    """Load pre-computed features"""
    features = {}
    
    # Morgan fingerprints
    if os.path.exists("data/processed/morgan_fingerprints.pkl"):
        with open("data/processed/morgan_fingerprints.pkl", 'rb') as f:
            features['morgan'] = pickle.load(f)
    
    # RDKit descriptors
    if os.path.exists("data/processed/rdkit_descriptors.pkl"):
        with open("data/processed/rdkit_descriptors.pkl", 'rb') as f:
            features['rdkit'] = pickle.load(f)
    
    return features

if __name__ == "__main__":
    df = load_dataset()
    features = load_features()
    print("Data preparation complete!")
