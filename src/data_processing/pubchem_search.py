import pubchempy as pcp
import pandas as pd
import time
from rdkit import Chem

def search_pubchem():
    """Search PubChem for specific ADC linker compounds"""
    
    # More specific ADC linker names and components
    pubchem_queries = [
        'Maleimidocaproic acid',
        'SPDB',
        'sulfo-SPDB', 
        'PEG4',
        'PEG8',
        'PEG12',
        'Glucuronide',
        'Val-Cit-PABC',
        'MC-VC-PABC',
        'Disulfide',
        'Biotin',
        'Succinimide',
        'Maleimide',
        'Citrulline',
        'Para-aminobenzyloxycarbonyl',
        'PABC'
    ]
    
    pubchem_data = []
    
    for query in pubchem_queries:
        print(f"Searching PubChem for: {query}")
        try:
            time.sleep(0.5)  # Be nice to PubChem API
            compounds = pcp.get_compounds(query, 'name')
            
            for compound in compounds:
                if compound.canonical_smiles:
                    # Validate with RDKit
                    mol = Chem.MolFromSmiles(compound.canonical_smiles)
                    if mol is not None:
                        pubchem_data.append({
                            'name': query,
                            'cid': compound.cid,
                            'canonical_smiles': compound.canonical_smiles,
                            'source': 'PubChem'
                        })
                        print(f"  Found: CID {compound.cid}")
                        
        except Exception as e:
            print(f"Error searching for {query}: {e}")
            continue
    
    pubchem_df = pd.DataFrame(pubchem_data)
    
    # Remove duplicates based on SMILES
    pubchem_df = pubchem_df.drop_duplicates(subset=['canonical_smiles'])
    
    print(f"Found {len(pubchem_df)} unique molecules from PubChem")
    
    # Save
    pubchem_df.to_csv('data/pubchem_linkers.csv', index=False)
    print("Saved PubChem data to data/pubchem_linkers.csv")
    
    return pubchem_df

if __name__ == "__main__":
    search_pubchem()
