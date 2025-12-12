import pandas as pd
import numpy as np
from rdkit import Chem

def fix_dataset():
    """Fix data type issues in the final dataset"""
    
    # Load the final dataset
    df = pd.read_csv('data/linkermind_final_dataset.csv')
    
    print("Original dataset info:")
    print(df.info())
    print(f"\nNull values in standard_smiles: {df['standard_smiles'].isnull().sum()}")
    
    # Fix the standard_smiles column - convert all to string and handle NaN values
    df['standard_smiles'] = df['standard_smiles'].astype(str)
    
    # Replace 'nan' strings with actual NaN
    df['standard_smiles'] = df['standard_smiles'].replace('nan', np.nan)
    
    # Remove rows with invalid SMILES
    original_count = len(df)
    df = df.dropna(subset=['standard_smiles'])
    
    # Remove any empty strings
    df = df[df['standard_smiles'].str.len() > 0]
    
    # Validate SMILES with RDKit
    valid_indices = []
    for idx, smiles in enumerate(df['standard_smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_indices.append(idx)
    
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    print(f"\nRemoved {original_count - len(df_valid)} invalid molecules")
    print(f"Final valid dataset: {len(df_valid)} molecules")
    
    # Rebalance the dataset
    positives = df_valid[df_valid['is_linker'] == 1]
    negatives = df_valid[df_valid['is_linker'] == 0]
    
    # Use the smaller class size
    min_class_size = min(len(positives), len(negatives))
    
    positives_balanced = positives.sample(n=min_class_size, random_state=42)
    negatives_balanced = negatives.sample(n=min_class_size, random_state=42)
    
    # Combine and shuffle
    final_df = pd.concat([positives_balanced, negatives_balanced], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset:")
    print(f"  Positive linkers: {len(final_df[final_df['is_linker'] == 1])}")
    print(f"  Negative decoys: {len(final_df[final_df['is_linker'] == 0])}")
    
    # Save the fixed dataset
    final_df.to_csv('data/linkermind_final_dataset_fixed.csv', index=False)
    print("\nSaved fixed dataset to data/linkermind_final_dataset_fixed.csv")
    
    return final_df

if __name__ == "__main__":
    fix_dataset()
