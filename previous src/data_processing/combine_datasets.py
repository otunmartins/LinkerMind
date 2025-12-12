import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
import os

def combine_all_data():
    """Combine ChEMBL, PubChem, and Literature data"""
    
    # Load all datasets
    chembl_df = pd.read_csv('data/chembl_extracted.csv')
    pubchem_df = pd.read_csv('data/pubchem_linkers.csv')
    literature_df = pd.read_csv('data/literature_linkers.csv')
    
    print(f"ChEMBL data: {len(chembl_df)} molecules")
    print(f"PubChem data: {len(pubchem_df)} molecules") 
    print(f"Literature data: {len(literature_df)} molecules")
    
    # Standardize column names for merging
    chembl_df = chembl_df[['molecule_chembl_id', 'pref_name', 'canonical_smiles', 'source']]
    pubchem_df = pubchem_df.rename(columns={'name': 'pref_name', 'cid': 'molecule_chembl_id'})
    pubchem_df = pubchem_df[['molecule_chembl_id', 'pref_name', 'canonical_smiles', 'source']]
    literature_df = literature_df[['molecule_chembl_id', 'pref_name', 'canonical_smiles', 'source']]
    
    # Combine all positive examples
    positive_df = pd.concat([chembl_df, pubchem_df, literature_df], ignore_index=True)
    print(f"Total positive examples before deduplication: {len(positive_df)}")
    
    # Remove exact duplicates based on SMILES
    positive_df = positive_df.drop_duplicates(subset=['canonical_smiles'])
    print(f"Total positive examples after deduplication: {len(positive_df)}")
    
    # Label as positive (linkers)
    positive_df['is_linker'] = 1
    
    return positive_df

def standardize_molecules(df):
    """Standardize all molecules using RDKit"""
    print("Standardizing molecules...")
    
    def standardize_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Remove salts
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol)
            
            # Generate canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return canonical_smiles
        except:
            return None
    
    df['standard_smiles'] = df['canonical_smiles'].apply(standardize_smiles)
    
    # Remove failed standardizations
    original_count = len(df)
    df = df.dropna(subset=['standard_smiles'])
    df = df.reset_index(drop=True)
    
    print(f"Removed {original_count - len(df)} molecules that failed standardization")
    return df

if __name__ == "__main__":
    # Combine all positive data
    positive_df = combine_all_data()
    
    # Standardize molecules
    positive_df = standardize_molecules(positive_df)
    
    print(f"Final positive dataset: {len(positive_df)} molecules")
    
    # Save combined positive data
    positive_df.to_csv('data/positive_linkers_combined.csv', index=False)
    print("Saved combined positive data to data/positive_linkers_combined.csv")
    
    # Show some statistics
    print("\n=== DATASET STATISTICS ===")
    source_counts = positive_df['source'].value_counts()
    print("Sources:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} molecules")
    
    # Show sample of molecules from each source
    print("\nSample molecules:")
    for source in positive_df['source'].unique():
        sample = positive_df[positive_df['source'] == source].head(2)
        print(f"\n{source}:")
        for _, row in sample.iterrows():
            print(f"  {row['pref_name']}: {row['standard_smiles'][:50]}...")
