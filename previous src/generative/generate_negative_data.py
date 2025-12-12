import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
import random
import os

def generate_negative_data(positive_df, num_decoys=None):
    """
    Generate negative examples (decoys) that are chemically plausible 
    but unlikely to be good ADC linkers
    """
    
    if num_decoys is None:
        num_decoys = len(positive_df)  # Match positive set size
    
    print(f"Generating {num_decoys} negative examples...")
    
    # Strategy 1: Get random drug-like molecules from ChEMBL that are NOT in our positive set
    print("Strategy 1: Getting random drug-like molecules from ChEMBL...")
    decoys_1 = get_random_chembl_decoys(positive_df, num_decoys // 2)
    
    # Strategy 2: Generate molecules with inappropriate properties for linkers
    print("Strategy 2: Generating molecules with inappropriate linker properties...")
    decoys_2 = generate_inappropriate_linkers(num_decoys // 4)
    
    # Strategy 3: Get molecules from specific non-linker classes
    print("Strategy 3: Getting molecules from non-linker classes...")
    decoys_3 = get_specific_non_linkers(num_decoys // 4)
    
    # Combine all decoys
    all_decoys = decoys_1 + decoys_2 + decoys_3
    
    # If we need more, generate simple fragments
    if len(all_decoys) < num_decoys:
        print(f"Generating additional {num_decoys - len(all_decoys)} simple fragments...")
        additional_decoys = generate_simple_fragments(num_decoys - len(all_decoys))
        all_decoys.extend(additional_decoys)
    
    # Remove any duplicates and trim to desired size
    all_decoys = list(set(all_decoys))[:num_decoys]
    
    print(f"Generated {len(all_decoys)} unique negative examples")
    
    # Create negative DataFrame
    negative_data = []
    for i, smiles in enumerate(all_decoys):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            negative_data.append({
                'molecule_chembl_id': f'DECOY_{i+1:05d}',
                'pref_name': f'Decoy_{i+1:05d}',
                'canonical_smiles': smiles,
                'standard_smiles': smiles,
                'source': 'Generated_Decoy',
                'is_linker': 0
            })
    
    negative_df = pd.DataFrame(negative_data)
    return negative_df

def get_random_chembl_decoys(positive_df, num_decoys):
    """Get random molecules from ChEMBL that are not in positive set"""
    from chembl_webresource_client.new_client import new_client
    import time
    
    molecule = new_client.molecule
    positive_smiles_set = set(positive_df['standard_smiles'].tolist())
    
    decoy_smiles = []
    
    # Search for random drug-like molecules
    search_terms = [
        'aspirin', 'ibuprofen', 'paracetamol', 'caffeine', 'glucose',
        'cholesterol', 'testosterone', 'estradiol', 'vitamin c', 'vitamin d',
        'morphine', 'penicillin', 'dopamine', 'serotonin', 'histamine'
    ]
    
    for term in search_terms:
        if len(decoy_smiles) >= num_decoys:
            break
            
        print(f"  Searching for: {term}")
        try:
            time.sleep(0.5)
            results = molecule.search(term)
            
            for result in results:
                if len(decoy_smiles) >= num_decoys:
                    break
                    
                try:
                    if ('molecule_structures' in result and 
                        result['molecule_structures'] and 
                        'canonical_smiles' in result['molecule_structures'] and 
                        result['molecule_structures']['canonical_smiles']):
                        
                        smiles = result['molecule_structures']['canonical_smiles']
                        mol = Chem.MolFromSmiles(smiles)
                        
                        # Basic filters to ensure drug-like but not linker-like
                        if (mol is not None and 
                            smiles not in positive_smiles_set and
                            Descriptors.MolWt(mol) < 600 and  # Smaller than typical linkers
                            Descriptors.MolLogP(mol) < 4):    # Reasonable lipophilicity
                            
                            decoy_smiles.append(smiles)
                            
                except:
                    continue
                    
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    return decoy_smiles

def generate_inappropriate_linkers(num_decoys):
    """Generate molecules with properties that make them bad linkers"""
    
    # These are molecules with properties that make them unsuitable as linkers
    bad_linker_smiles = [
        # Too flexible (long chains without rigidity)
        "CCCCCCCCCCCCCCCC",  # Very long alkane
        "C(COCCOCCOCCOCCOCCO)CO",  # Very long PEG
        "C1CCCCCCCCCC1",  # Large ring - too flexible
        
        # Too hydrophobic
        "CCCCCCCCCCCC",  # Dodecane
        "CC(C)CCCC(C)C",  # Hydrocarbon
        "ClC1=CC=CC=C1",  # Chlorobenzene
        
        # Too hydrophilic/charged (poor membrane permeability)
        "C([C@@H](C(=O)O)N)C(=O)O",  # Aspartic acid
        "C(CNCCNCCNCCN)CO",  # Polyamine
        "NCCCCCCCCCCN",  # Diamine
        
        # Too reactive/unstable
        "CCOC(=O)C=CC(=O)OCC",  # Diethyl maleate (too reactive)
        "O=C1C=CC=CN1",  # Reactive heterocycle
        "C1=CN=CC=C1",  # Pyridine (can coordinate metals)
        
        # Too large/bulky
        "CC1(C2CCC1(CC2)C3=CC=CC=C3)C4=CC=CC=C4",  # Very bulky
        "C1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3",  # Triphenyl
    ]
    
    return bad_linker_smiles[:num_decoys]

def get_specific_non_linkers(num_decoys):
    """Get molecules from classes that are typically not used as linkers"""
    
    non_linker_classes = [
        # Steroids
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        
        # Sugars
        "C(C1C(C(C(C(O1)O)O)O)O)O",
        "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
        
        # Nucleotides
        "C1=NC2=NC=NC(=C2N1)N",
        "C1C(C(OC1N2C=NC3=C2N=CN=C3N)O)O",
        
        # Fatty acids
        "CCCCCCCCCCCCCCCC(=O)O",
        "CCCCCCCC=CCCCCCCCC(=O)O",
        
        # Complex natural products
        "CN1C2CCC1CC(C2)OC(=O)C(CO)c1ccccc1",
    ]
    
    return non_linker_classes[:num_decoys]

def generate_simple_fragments(num_decoys):
    """Generate simple molecular fragments"""
    
    simple_fragments = [
        "CCO", "CCN", "CC(=O)O", "CCN(CC)CC", "C1CCCCC1",
        "C1CCCC1", "C1=CC=CC=C1", "C1CC2CCCC2C1", "CC(C)C",
        "CCOC(=O)C", "CC(C)OC(=O)C", "C1COCCO1", "C1CCCCC1",
        "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC"
    ]
    
    return simple_fragments[:num_decoys]

def create_final_dataset(positive_df, negative_df):
    """Create the final balanced dataset"""
    
    print("Creating final dataset...")
    
    # Select the same number of positive and negative examples for balance
    num_positives = len(positive_df)
    num_negatives = len(negative_df)
    
    # Use the smaller set to determine final size
    final_size = min(num_positives, num_negatives)
    
    # Sample equal numbers from both classes
    positive_final = positive_df.sample(n=final_size, random_state=42)
    negative_final = negative_df.sample(n=final_size, random_state=42)
    
    # Combine and shuffle
    final_df = pd.concat([positive_final, negative_final], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final balanced dataset: {len(final_df)} molecules")
    print(f"  - Positive linkers: {len(final_df[final_df['is_linker'] == 1])}")
    print(f"  - Negative decoys: {len(final_df[final_df['is_linker'] == 0])}")
    
    return final_df

if __name__ == "__main__":
    # Load positive data
    positive_df = pd.read_csv('data/positive_linkers_combined.csv')
    
    # Generate negative data
    negative_df = generate_negative_data(positive_df)
    
    # Save negative data
    negative_df.to_csv('data/negative_decoys.csv', index=False)
    print("Saved negative data to data/negative_decoys.csv")
    
    # Create final balanced dataset
    final_df = create_final_dataset(positive_df, negative_df)
    
    # Save final dataset
    final_df.to_csv('data/linkermind_final_dataset.csv', index=False)
    print("Saved final dataset to data/linkermind_final_dataset.csv")
    
    # Print final statistics
    print("\n=== FINAL DATASET COMPOSITION ===")
    source_counts = final_df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    # Basic molecular statistics
    print("\n=== MOLECULAR PROPERTIES ===")
    from rdkit.Chem import Descriptors
    
    pos_mols = [Chem.MolFromSmiles(s) for s in final_df[final_df['is_linker'] == 1]['standard_smiles']]
    neg_mols = [Chem.MolFromSmiles(s) for s in final_df[final_df['is_linker'] == 0]['standard_smiles']]
    
    pos_mols = [m for m in pos_mols if m is not None]
    neg_mols = [m for m in neg_mols if m is not None]
    
    def get_stats(mols, label):
        if not mols:
            return
        mw = [Descriptors.MolWt(m) for m in mols]
        logp = [Descriptors.MolLogP(m) for m in mols]
        print(f"{label}:")
        print(f"  Molecular Weight: {sum(mw)/len(mw):.1f} ± {max(mw)-min(mw):.1f}")
        print(f"  LogP: {sum(logp)/len(logp):.2f} ± {max(logp)-min(logp):.2f}")
        print(f"  Count: {len(mols)}")
    
    get_stats(pos_mols, "Positive Linkers")
    get_stats(neg_mols, "Negative Decoys")
