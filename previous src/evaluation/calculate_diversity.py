import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

def calculate_diversity_prep(df):
    """Prepare data for diversity calculation"""
    properties = []
    
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['standard_smiles'])
        if mol is not None:
            props = {
                'is_linker': row['is_linker'],
                'source': row['source'],
                'smiles': row['standard_smiles'],
                'mw': AllChem.CalcExactMolWt(mol),
            }
            properties.append(props)
    
    return pd.DataFrame(properties)

def calculate_diversity(props_df):
    """Calculate chemical diversity of the dataset"""
    
    print("\n=== CHEMICAL DIVERSITY ANALYSIS ===")
    
    # Generate fingerprints for diversity calculation
    fingerprints = []
    valid_smiles = []
    
    for _, row in props_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp)
            valid_smiles.append(row['smiles'])
    
    if len(fingerprints) < 2:
        print("Not enough valid molecules for diversity calculation")
        return
    
    # Calculate pairwise Tanimoto similarities
    similarity_matrix = []
    for i in range(len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        similarity_matrix.extend(similarities)
    
    avg_similarity = np.mean(similarity_matrix) if similarity_matrix else 0
    print(f"Average pairwise Tanimoto similarity: {avg_similarity:.3f}")
    
    # Calculate diversity by class
    linker_indices = [i for i, label in enumerate(props_df['is_linker']) if label == 1]
    decoy_indices = [i for i, label in enumerate(props_df['is_linker']) if label == 0]
    
    linker_fps = [fingerprints[i] for i in linker_indices if i < len(fingerprints)]
    decoy_fps = [fingerprints[i] for i in decoy_indices if i < len(fingerprints)]
    
    if len(linker_fps) > 1:
        linker_similarities = []
        for i in range(len(linker_fps)):
            similarities = DataStructs.BulkTanimotoSimilarity(linker_fps[i], linker_fps[:i])
            linker_similarities.extend(similarities)
        avg_linker_sim = np.mean(linker_similarities) if linker_similarities else 0
        print(f"Linker diversity (avg similarity): {avg_linker_sim:.3f}")
    
    if len(decoy_fps) > 1:
        decoy_similarities = []
        for i in range(len(decoy_fps)):
            similarities = DataStructs.BulkTanimotoSimilarity(decoy_fps[i], decoy_fps[:i])
            decoy_similarities.extend(similarities)
        avg_decoy_sim = np.mean(decoy_similarities) if decoy_similarities else 0
        print(f"Decoy diversity (avg similarity): {avg_decoy_sim:.3f}")

if __name__ == "__main__":
    df = pd.read_csv('data/linkermind_final_dataset_fixed.csv')
    props_df = calculate_diversity_prep(df)
    calculate_diversity(props_df)
