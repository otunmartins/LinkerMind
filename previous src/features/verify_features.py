import pickle
import numpy as np
import pandas as pd

def verify_features():
    """Verify the generated features"""
    
    print("=== VERIFYING GENERATED FEATURES ===\n")
    
    # Check Morgan fingerprints
    try:
        with open('data/features/morgan_fingerprints.pkl', 'rb') as f:
            morgan_data = pickle.load(f)
        
        print("✅ Morgan Fingerprints:")
        print(f"   Shape: {morgan_data['features'].shape}")
        print(f"   Molecules: {len(morgan_data['smiles'])}")
        print(f"   Labels: {sum(morgan_data['labels'])} linkers, {len(morgan_data['labels']) - sum(morgan_data['labels'])} decoys")
        print(f"   Data type: {morgan_data['features'].dtype}")
        print(f"   Memory usage: {morgan_data['features'].nbytes / 1024 / 1024:.2f} MB")
        
    except FileNotFoundError:
        print("❌ Morgan fingerprints file not found")
    
    # Check if RDKit descriptors exist
    try:
        with open('data/features/rdkit_descriptors.pkl', 'rb') as f:
            rdkit_data = pickle.load(f)
        
        print("\n✅ RDKit Descriptors:")
        print(f"   Shape: {rdkit_data['features'].shape}")
        print(f"   Descriptors: {rdkit_data['descriptor_names']}")
        
    except FileNotFoundError:
        print("\n❌ RDKit descriptors file not found (still processing)")
    
    # Check if graph data exists
    try:
        with open('data/features/graph_data.pkl', 'rb') as f:
            graph_data = pickle.load(f)
        
        print("\n✅ Graph Data:")
        print(f"   Graphs: {len(graph_data['graph_data'])}")
        print(f"   Sample node features shape: {graph_data['graph_data'][0].x.shape}")
        print(f"   Sample edge features shape: {graph_data['graph_data'][0].edge_attr.shape}")
        
    except FileNotFoundError:
        print("\n❌ Graph data file not found (still processing)")

if __name__ == "__main__":
    verify_features()
