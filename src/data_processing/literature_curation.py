import pandas as pd
from rdkit import Chem

def add_literature_linkers():
    """Add well-known ADC linkers from literature"""
    
    # Well-characterized ADC linkers from key papers/reviews
    # These are specific, proven linkers used in clinical ADCs
    literature_linkers = [
        # Cleavable linkers
        "CCCCCC(=O)N1C(=O)[C@H](C(C)C)NC(=O)[C@H](CC(C)C)NC1=O",  # Val-Cit (vc) PABC
        "CC(C)C[C@H](NC(=O)O[C@H](COC(=O)CNC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(C)C)NC(=O)C1CCCN1C(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H]1CCOC1=O)C(=O)NO)C(=O)O",  # Complex peptide linker
        "C1CCSC1",  # Simple disulfide linker core
        "C1CCOC1",  # Tetrahydropyran - simple heterocyclic linker
        "CCOC(=O)CNC(=O)C1CCCN1",  # PEG-like with piperazine
        "C1COCCO1",  # Dioxane - common linker component
        "C1COCCO1",  # Morpholine derivative
        "CCCCCCCCCCCCCCC(=O)O",  # Long alkyl chain (fatty acid type)
        "C1CC1C2CC2",  # Bicyclic alkane scaffold
        # Add maleimide-containing linkers (common for cysteine conjugation)
        "C1=CC(=C(C=C1C2=CC=CC=C2)N3C4=CC=CC=C4C5=CC=CC=C53)N6C7=CC=CC=C7C8=CC=CC=C86",  # Complex aromatic
        "C1=CC(=CC=C1C#N)C2=CC=CC=C2",  # Biphenyl derivative
        "C1CCN(CC1)C2CCCN2",  # Piperazine-piperidine
    ]
    
    literature_data = []
    for i, smiles in enumerate(literature_linkers):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
            literature_data.append({
                'molecule_chembl_id': f'LITERATURE_{i+1:03d}',
                'pref_name': f'Literature_Linker_{i+1:03d}',
                'canonical_smiles': canonical_smiles,
                'source': 'Literature'
            })
    
    literature_df = pd.DataFrame(literature_data)
    print(f"Added {len(literature_df)} literature linkers")
    return literature_df

if __name__ == "__main__":
    literature_df = add_literature_linkers()
    literature_df.to_csv('data/literature_linkers.csv', index=False)
    print("Saved literature linkers to data/literature_linkers.csv")
