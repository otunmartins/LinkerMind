import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import DataStructs
import torch
from torch_geometric.data import Data, Dataset
import os
from tqdm import tqdm
import pickle

class MolecularFeatureEngineer:
    """Engineer molecular features for deep learning"""
    
    def __init__(self):
        self.descriptor_names = []
    
    def compute_morgan_fingerprints(self, smiles_list, radius=2, n_bits=2048):
        """Compute Morgan fingerprints (ECFP-like)"""
        print("Computing Morgan fingerprints...")
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints.append(arr)
                valid_indices.append(i)
        
        return np.array(fingerprints), valid_indices
    
    def compute_rdkit_descriptors(self, smiles_list):
        """Compute RDKit molecular descriptors"""
        print("Computing RDKit descriptors...")
        
        # List of important descriptors for drug discovery - CORRECTED NAMES
        descriptor_funcs = [
            Descriptors.MolWt,
            Descriptors.MolLogP,
            Descriptors.NumHDonors,
            Descriptors.NumHAcceptors,
            Descriptors.NumRotatableBonds,
            Descriptors.NumAromaticRings,
            Descriptors.TPSA,
            Descriptors.FractionCSP3,  # CORRECTED: was FractionCsp3
            Descriptors.NumHeteroatoms,
            rdMolDescriptors.CalcNumRings,
            rdMolDescriptors.CalcNumAliphaticRings,
            rdMolDescriptors.CalcNumAromaticRings,
            rdMolDescriptors.CalcNumAmideBonds,
        ]
        
        self.descriptor_names = [func.__name__ for func in descriptor_funcs]
        descriptors = []
        valid_indices = []
        
        for i, smiles in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc_values = []
                for func in descriptor_funcs:
                    try:
                        value = func(mol)
                        # Handle potential NaN/inf values
                        if np.isnan(value) or np.isinf(value):
                            value = 0.0
                        desc_values.append(float(value))
                    except:
                        desc_values.append(0.0)
                descriptors.append(desc_values)
                valid_indices.append(i)
        
        return np.array(descriptors), valid_indices
    
    def create_graph_data(self, smiles_list, labels):
        """Create graph representations for GNNs"""
        print("Creating graph representations...")
        
        graph_data_list = []
        valid_indices = []
        
        for i, (smiles, label) in enumerate(tqdm(zip(smiles_list, labels), total=len(smiles_list))):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    # Node features: atom features
                    node_features = self._get_node_features(mol)
                    
                    # Edge features: bond features  
                    edge_features, edge_index = self._get_edge_features(mol)
                    
                    # Graph label
                    y = torch.tensor([label], dtype=torch.float)
                    
                    # Create PyTorch Geometric Data object
                    graph_data = Data(
                        x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        y=y,
                        smiles=smiles
                    )
                    
                    graph_data_list.append(graph_data)
                    valid_indices.append(i)
                    
                except Exception as e:
                    print(f"Error processing molecule {i}: {e}")
                    continue
        
        return graph_data_list, valid_indices
    
    def _get_node_features(self, mol):
        """Get atom-level features for GNN"""
        periodic_table = Chem.GetPeriodicTable()
        all_node_features = []
        
        for atom in mol.GetAtoms():
            node_features = []
            
            # Basic atom features
            node_features.append(atom.GetAtomicNum())  # Atomic number
            node_features.append(atom.GetDegree())     # Degree
            node_features.append(atom.GetTotalNumHs()) # Number of H
            node_features.append(atom.GetFormalCharge()) # Formal charge
            node_features.append(int(atom.GetIsAromatic())) # Aromaticity
            node_features.append(atom.GetHybridization().real) # Hybridization
            node_features.append(atom.GetMass())       # Atomic mass
            node_features.append(periodic_table.GetNOuterElecs(atom.GetAtomicNum())) # Valence electrons
            
            # One-hot encoding for common elements
            common_elements = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
            for element in common_elements:
                node_features.append(1 if atom.GetAtomicNum() == element else 0)
            
            # One-hot encoding for hybridization
            hybridizations = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, 
                            Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D]
            for hyb in hybridizations:
                node_features.append(1 if atom.GetHybridization() == hyb else 0)
            
            all_node_features.append(node_features)
        
        return torch.tensor(all_node_features, dtype=torch.float)
    
    def _get_edge_features(self, mol):
        """Get bond-level features for GNN"""
        edge_features = []
        edge_index = []
        
        for bond in mol.GetBonds():
            # Get bond features
            bond_features = []
            bond_features.append(bond.GetBondTypeAsDouble())  # Bond type
            bond_features.append(int(bond.GetIsConjugated())) # Conjugation
            bond_features.append(int(bond.IsInRing()))        # In ring
            bond_features.append(bond.GetStereo().real)       # Stereo
            
            edge_features.append(bond_features)
            edge_features.append(bond_features)  # Add reverse direction
            
            # Add edges in both directions for undirected graph
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        
        if edge_features:  # Check if we have any bonds
            edge_features = torch.tensor(edge_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # Handle molecules with no bonds (single atoms)
            edge_features = torch.tensor([], dtype=torch.float)
            edge_index = torch.tensor([], dtype=torch.long)
        
        return edge_features, edge_index

def main():
    """Main function for feature engineering"""
    
    # Load the fixed dataset
    df = pd.read_csv('data/linkermind_final_dataset_fixed.csv')
    print(f"Loaded dataset with {len(df)} molecules")
    
    # Initialize feature engineer
    engineer = MolecularFeatureEngineer()
    
    # Prepare data
    smiles_list = df['standard_smiles'].tolist()
    labels = df['is_linker'].tolist()
    
    # 1. Compute Morgan Fingerprints (Baseline) - Already done, but we'll reload to be safe
    print("\n" + "="*50)
    print("1. COMPUTING MORGAN FINGERPRINTS")
    print("="*50)
    morgan_fps, morgan_indices = engineer.compute_morgan_fingerprints(smiles_list)
    print(f"Morgan fingerprints shape: {morgan_fps.shape}")
    
    # 2. Compute RDKit Descriptors - FIXED
    print("\n" + "="*50)
    print("2. COMPUTING RDKIT DESCRIPTORS")
    print("="*50)
    rdkit_descriptors, desc_indices = engineer.compute_rdkit_descriptors(smiles_list)
    print(f"RDKit descriptors shape: {rdkit_descriptors.shape}")
    print(f"Descriptor names: {engineer.descriptor_names}")
    
    # 3. Create Graph Data for GNNs
    print("\n" + "="*50)
    print("3. CREATING GRAPH REPRESENTATIONS")
    print("="*50)
    graph_data_list, graph_indices = engineer.create_graph_data(smiles_list, labels)
    print(f"Created {len(graph_data_list)} graph representations")
    
    # Save all features
    print("\n" + "="*50)
    print("4. SAVING FEATURES")
    print("="*50)
    
    # Create features directory
    os.makedirs('data/features', exist_ok=True)
    
    # Save Morgan fingerprints
    with open('data/features/morgan_fingerprints.pkl', 'wb') as f:
        pickle.dump({
            'features': morgan_fps,
            'indices': morgan_indices,
            'smiles': [smiles_list[i] for i in morgan_indices],
            'labels': [labels[i] for i in morgan_indices]
        }, f)
    
    # Save RDKit descriptors
    with open('data/features/rdkit_descriptors.pkl', 'wb') as f:
        pickle.dump({
            'features': rdkit_descriptors,
            'indices': desc_indices,
            'descriptor_names': engineer.descriptor_names,
            'smiles': [smiles_list[i] for i in desc_indices],
            'labels': [labels[i] for i in desc_indices]
        }, f)
    
    # Save graph data
    with open('data/features/graph_data.pkl', 'wb') as f:
        pickle.dump({
            'graph_data': graph_data_list,
            'indices': graph_indices,
            'smiles': [smiles_list[i] for i in graph_indices],
            'labels': [labels[i] for i in graph_indices]
        }, f)
    
    print("Saved all features to data/features/ directory")
    
    # Print summary
    print("\n" + "="*50)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*50)
    print(f"Morgan Fingerprints: {morgan_fps.shape}")
    print(f"RDKit Descriptors: {rdkit_descriptors.shape}")
    print(f"Graph Representations: {len(graph_data_list)} molecules")
    print(f"Descriptor Names: {engineer.descriptor_names}")

if __name__ == "__main__":
    main()
