import pickle
import pandas as pd
from rdkit import Chem
import os

def extract_chembl_data():
    """Extract and process ChEMBL results"""
    
    # Load the saved results
    try:
        with open('data/raw_chembl_results.pkl', 'rb') as f:
            all_results = pickle.load(f)
    except FileNotFoundError:
        print("Error: Run chembl_search.py first!")
        return
    
    print(f"Processing {len(all_results)} raw results...")
    
    data_list = []
    skipped_count = 0
    
    for i, result in enumerate(all_results):
        if i % 100 == 0:
            print(f"Processing result {i}/{len(all_results)}...")
            
        try:
            # Check if molecule_structures exists and has canonical SMILES
            if ('molecule_structures' in result and 
                result['molecule_structures'] and 
                'canonical_smiles' in result['molecule_structures'] and 
                result['molecule_structures']['canonical_smiles']):
                
                smiles = result['molecule_structures']['canonical_smiles']
                molecule_chembl_id = result['molecule_chembl_id']
                pref_name = result.get('pref_name', 'N/A')
                
                # Basic SMILES validation with RDKit
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    data_list.append({
                        'molecule_chembl_id': molecule_chembl_id,
                        'pref_name': pref_name,
                        'canonical_smiles': smiles,
                        'source': 'ChEMBL'
                    })
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing result {i}: {e}")
            skipped_count += 1
            continue
    
    # Create DataFrame
    chembl_df = pd.DataFrame(data_list)
    
    print(f"Successfully extracted {len(chembl_df)} molecules with valid SMILES.")
    print(f"Skipped {skipped_count} invalid entries.")
    
    # Save to CSV
    chembl_df.to_csv('data/chembl_extracted.csv', index=False)
    print("Saved extracted data to data/chembl_extracted.csv")
    
    # Show some basic stats
    if len(chembl_df) > 0:
        print("\nSample of extracted data:")
        print(chembl_df.head())
        
        # Show unique names to verify we're getting linker-related compounds
        print(f"\nUnique names sample: {chembl_df['pref_name'].unique()[:10]}")
    
    return chembl_df

if __name__ == "__main__":
    extract_chembl_data()
