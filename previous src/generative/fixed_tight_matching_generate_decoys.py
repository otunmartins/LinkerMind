# fixed_tight_matching_generate_decoys.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

def calculate_molecular_properties(smiles):
    """Calculate key molecular properties for matching"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        return {
            'mw': mw,
            'logp': logp, 
            'tpsa': tpsa,
            'hbd': hbd,
            'hba': hba,
            'rotatable_bonds': rotatable_bonds,
            'aromatic_rings': aromatic_rings
        }
    except:
        return None

def generate_tight_matched_decoys(positive_smiles, num_decoys_per_positive=1, property_tolerance=0.2):
    """
    Generate decoys with TIGHT property matching
    property_tolerance: 0.2 = 20% tolerance window for each property
    """
    print("🔬 Generating tightly matched decoys...")
    
    # Common small molecules and fragments for decoy generation
    decoy_library = [
        # Small heterocycles
        'c1ccncc1', 'c1ccoc1', 'c1ccsc1', 'c1cncnc1', 'c1c[nH]cc1',
        # Aliphatic chains and fragments  
        'CCO', 'CCN', 'CC(=O)O', 'CCN(CC)CC', 'C1CCCCC1', 'C1CCCC1',
        # Common medicinal chemistry fragments
        'c1ccc(C(=O)O)cc1', 'c1ccc(C(=O)N)cc1', 'c1ccc(CN)cc1', 
        'c1ccc(OC)cc1', 'c1ccc(Cl)cc1', 'c1ccc(Br)cc1', 'c1ccc(I)cc1',
        'c1ccc(F)cc1', 'c1ccccc1C(=O)O', 'c1ccccc1C(=O)N',
        # Nitrogen-containing
        'CN1C=NC=N1', 'C1COCCO1', 'C1CCNCC1', 'C1CNCCN1',
        # Oxygen-containing
        'C1CCOC1', 'C1COCCO1', 'CCOC(=O)C', 'CC(=O)NC',
        # Sulfur-containing  
        'CS(=O)(=O)C', 'CSC', 'CCSCC'
    ]
    
    # Expand decoy library by combining fragments
    expanded_decoys = []
    for i, decoy1 in enumerate(decoy_library):
        for j, decoy2 in enumerate(decoy_library):
            if i != j:
                # Try different combinations
                expanded_decoys.append(f"{decoy1}.{decoy2}")
                expanded_decoys.append(f"{decoy1}{decoy2}")
    
    decoy_library += expanded_decoys
    print(f"📚 Using decoy library of {len(decoy_library)} molecules")
    
    matched_decoys = []
    positive_properties = []
    
    # Calculate properties for all positives
    print("📊 Calculating positive linker properties...")
    for smile in tqdm(positive_smiles):
        props = calculate_molecular_properties(smile)
        if props:
            positive_properties.append((smile, props))
    
    print(f"✅ Calculated properties for {len(positive_properties)} positive linkers")
    
    # For each positive, find best-matching decoy
    print("🎯 Matching decoys to positives...")
    for pos_smile, pos_props in tqdm(positive_properties):
        best_decoy = None
        best_score = float('inf')
        
        for decoy_smile in decoy_library:
            decoy_props = calculate_molecular_properties(decoy_smile)
            if not decoy_props:
                continue
                
            # Calculate property matching score (lower = better match)
            mw_diff = abs(pos_props['mw'] - decoy_props['mw']) / pos_props['mw']
            logp_diff = abs(pos_props['logp'] - decoy_props['logp']) / (abs(pos_props['logp']) + 1)
            tpsa_diff = abs(pos_props['tpsa'] - decoy_props['tpsa']) / (pos_props['tpsa'] + 1)
            
            # Only consider if within tolerance
            if (mw_diff <= property_tolerance and 
                logp_diff <= property_tolerance and 
                tpsa_diff <= property_tolerance):
                
                total_score = mw_diff + logp_diff + tpsa_diff
                if total_score < best_score:
                    best_score = total_score
                    best_decoy = decoy_smile
        
        if best_decoy:
            matched_decoys.append(best_decoy)
        else:
            # If no good match found, use a random decoy (fallback)
            matched_decoys.append(np.random.choice(decoy_library))
    
    return matched_decoys

def main():
    print("=== Creating TIGHTLY MATCHED Balanced Dataset ===\n")
    
    # Load your positive linkers
    # Assuming you have a list of positive linker SMILES
    positive_df = pd.read_csv('positive_linkers_combined.csv')
    positive_smiles = positive_df['canonical_smiles'].dropna().unique().tolist()
    
    print(f"📊 Starting with {len(positive_smiles)} positive linkers")
    
    # Generate tightly matched decoys
    decoy_smiles = generate_tight_matched_decoys(positive_smiles, property_tolerance=0.3)
    
    # Create balanced dataset
    positive_data = [{'smiles': smi, 'is_linker': 1} for smi in positive_smiles]
    negative_data = [{'smiles': smi, 'is_linker': 0} for smi in decoy_smiles]
    
    balanced_df = pd.DataFrame(positive_data + negative_data)
    
    # Save the dataset
    balanced_df.to_csv('linkermind_tightly_balanced_dataset.csv', index=False)
    print(f"\n💾 Saved TIGHTLY balanced dataset to linkermind_tightly_balanced_dataset.csv")
    print(f"📦 Final dataset: {len(balanced_df)} molecules")
    print(f"   - {len(positive_smiles)} positives")
    print(f"   - {len(decoy_smiles)} negatives")
    
    # Analyze property balance
    print("\n=== 📈 PROPERTY BALANCE VERIFICATION ===")
    pos_props = []
    neg_props = []
    
    for smi in positive_smiles[:100]:  # Sample 100
        props = calculate_molecular_properties(smi)
        if props: pos_props.append(props)
    
    for smi in decoy_smiles[:100]:  # Sample 100  
        props = calculate_molecular_properties(smi)
        if props: neg_props.append(props)
    
    if pos_props and neg_props:
        pos_mw = np.mean([p['mw'] for p in pos_props])
        neg_mw = np.mean([p['mw'] for p in neg_props])
        pos_tpsa = np.mean([p['tpsa'] for p in pos_props])
        neg_tpsa = np.mean([p['tpsa'] for p in neg_props])
        
        print(f"Molecular Weight:")
        print(f"  Positives: {pos_mw:.1f} ± {np.std([p['mw'] for p in pos_props]):.1f}")
        print(f"  Negatives: {neg_mw:.1f} ± {np.std([p['mw'] for p in neg_props]):.1f}")
        print(f"  Difference: {abs(pos_mw - neg_mw):.1f} Da")
        
        print(f"TPSA:")
        print(f"  Positives: {pos_tpsa:.1f} ± {np.std([p['tpsa'] for p in pos_props]):.1f}")
        print(f"  Negatives: {neg_tpsa:.1f} ± {np.std([p['tpsa'] for p in neg_props]):.1f}")
        print(f"  Difference: {abs(pos_tpsa - neg_tpsa):.1f}")
        
        if abs(pos_mw - neg_mw) < 100:  # Target: < 100 Da difference
            print("✅ EXCELLENT: MW balance achieved!")
        else:
            print("⚠️  Needs improvement: MW difference still too large")

if __name__ == "__main__":
    main()
