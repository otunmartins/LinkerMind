# fast_property_balanced_dataset.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import DataStructs
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import random

def calculate_molecular_properties(smiles):
    """Calculate key molecular properties efficiently"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms()
        }
    except:
        return None

def load_large_decoy_library():
    """Load a large library of potential decoys from public sources"""
    print("📚 Loading large decoy library...")
    
    # Common small molecules, fragments, and drug-like compounds
    decoy_smiles = [
        # Small heterocycles
        'c1ccncc1', 'c1ccoc1', 'c1ccsc1', 'c1cncnc1', 'c1c[nH]cc1', 'c1c[nH]nc1',
        'c1ccccc1', 'C1CCCCC1', 'C1CCCC1', 'c1ccc2ccccc2c1', 
        
        # Aliphatic chains and functional groups
        'CCO', 'CCN', 'CC(=O)O', 'CCN(CC)CC', 'CCOC(=O)C', 'CC(=O)N',
        'CC(C)C', 'CC(C)(C)C', 'C(C)(C)C', 'CCC', 'CCCC', 'CCCCC',
        
        # Common medicinal chemistry fragments
        'c1ccc(C(=O)O)cc1', 'c1ccc(C(=O)N)cc1', 'c1ccc(CN)cc1', 'c1ccc(OC)cc1',
        'c1ccc(Cl)cc1', 'c1ccc(Br)cc1', 'c1ccc(I)cc1', 'c1ccc(F)cc1', 
        'c1ccccc1C(=O)O', 'c1ccccc1C(=O)N', 'c1ccccc1OC', 'c1ccccc1CN',
        
        # Nitrogen-containing
        'CN1C=NC=N1', 'C1COCCO1', 'C1CCNCC1', 'C1CNCCN1', 'NCCN', 'NCCO',
        'CNC', 'CN(C)C', 'CCN', 'CCCN',
        
        # Oxygen-containing
        'C1CCOC1', 'C1COCCO1', 'CCOC(=O)C', 'CC(=O)NC', 'CCOCC', 'CCCO',
        'CCO', 'CCCOC',
        
        # Sulfur-containing  
        'CS(=O)(=O)C', 'CSC', 'CCSCC', 'CS(C)C', 'CCS(=O)(=O)C',
        
        # Mixed heteroatoms
        'c1ncccc1', 'c1cnccc1', 'c1cnccn1', 'c1nc[nH]c1', 'c1ncsc1',
        'O=C1CCCC1', 'O=C1CCCCN1', 'O=C1CCCCO1',
        
        # Additional diverse fragments
        'Cc1ccccc1', 'Fc1ccccc1', 'Clc1ccccc1', 'Brc1ccccc1', 
        'Ic1ccccc1', 'C#Cc1ccccc1', 'C=Cc1ccccc1',
        'CC(=O)c1ccccc1', 'OCc1ccccc1', 'COc1ccccc1', 'CNc1ccccc1',
        'CCOC(=O)c1ccccc1', 'CCNC(=O)c1ccccc1',
        
        # Aliphatic heterocycles
        'C1CCOC1', 'C1CCSC1', 'C1CCNC1', 'C1COCCN1', 'C1CNCCN1',
        'O1CCOC1', 'O1CCNC1', 'S1CCNC1',
        
        # Common building blocks
        'CC(C)OC(=O)N', 'CC(C)NC(=O)O', 'CCOC(=O)CN', 'CCNC(=O)CO',
        'O=C(N)CCN', 'O=C(O)CCN', 'NCC(=O)O', 'NCC(=O)N',
    ]
    
    # Expand by combining fragments (much more efficient)
    print("🔄 Expanding decoy library by combining fragments...")
    base_fragments = decoy_smiles[:30]  # Use first 30 for combinations
    for i in range(len(base_fragments)):
        for j in range(i+1, len(base_fragments)):
            # Try different connection types
            decoy_smiles.append(f"{base_fragments[i]}.{base_fragments[j]}")
            decoy_smiles.append(f"{base_fragments[i]}{base_fragments[j]}")
            if len(decoy_smiles) > 10000:  # Limit size
                break
        if len(decoy_smiles) > 10000:
            break
    
    # Remove duplicates and invalid SMILES
    decoy_smiles = list(set(decoy_smiles))
    valid_decoys = []
    
    print("⚗️ Validating decoy molecules...")
    for smi in tqdm(decoy_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_decoys.append(smi)
    
    print(f"✅ Loaded {len(valid_decoys)} valid decoy molecules")
    return valid_decoys

def create_property_balanced_dataset(positive_smiles, decoy_library, target_ratio=1.0):
    """Create balanced dataset using property binning"""
    print("🎯 Creating property-balanced dataset...")
    
    # Calculate properties for positives
    print("📊 Analyzing positive linkers...")
    positive_data = []
    for smi in tqdm(positive_smiles):
        props = calculate_molecular_properties(smi)
        if props:
            positive_data.append({'smiles': smi, 'is_linker': 1, **props})
    
    print("📊 Analyzing decoy library...")
    decoy_data = []
    for smi in tqdm(decoy_library):
        props = calculate_molecular_properties(smi)
        if props:
            decoy_data.append({'smiles': smi, 'is_linker': 0, **props})
    
    # Create property bins for matching
    def create_property_bins(data, prop_name, num_bins=10):
        values = [d[prop_name] for d in data]
        return np.quantile(values, np.linspace(0, 1, num_bins + 1))
    
    # Sample decoys to match positive distribution
    print("📈 Matching property distributions...")
    
    # Use simpler approach: sample decoys with similar MW range
    pos_mw_values = [d['mw'] for d in positive_data]
    pos_tpsa_values = [d['tpsa'] for d in positive_data]
    
    mw_range = (np.percentile(pos_mw_values, 10), np.percentile(pos_mw_values, 90))
    tpsa_range = (np.percentile(pos_tpsa_values, 10), np.percentile(pos_tpsa_values, 90))
    
    print(f"🎯 Target property ranges:")
    print(f"   MW: {mw_range[0]:.1f} - {mw_range[1]:.1f}")
    print(f"   TPSA: {tpsa_range[0]:.1f} - {tpsa_range[1]:.1f}")
    
    # Filter decoys within property ranges
    matched_decoys = []
    for decoy in decoy_data:
        if (mw_range[0] <= decoy['mw'] <= mw_range[1] and 
            tpsa_range[0] <= decoy['tpsa'] <= tpsa_range[1]):
            matched_decoys.append(decoy)
    
    print(f"📦 Found {len(matched_decoys)} decoys within property ranges")
    
    # If not enough decoys, relax criteria
    if len(matched_decoys) < len(positive_data):
        print("⚠️  Not enough matching decoys, relaxing criteria...")
        mw_range = (np.min(pos_mw_values), np.max(pos_mw_values))
        tpsa_range = (np.min(pos_tpsa_values), np.max(pos_tpsa_values))
        
        matched_decoys = []
        for decoy in decoy_data:
            if (mw_range[0] <= decoy['mw'] <= mw_range[1] and 
                tpsa_range[0] <= decoy['tpsa'] <= tpsa_range[1]):
                matched_decoys.append(decoy)
    
    # Sample the required number of decoys
    num_decoys_needed = min(len(positive_data), len(matched_decoys))
    selected_decoys = random.sample(matched_decoys, num_decoys_needed)
    
    # Combine datasets
    balanced_data = positive_data + selected_decoys
    random.shuffle(balanced_data)
    
    return balanced_data

def main():
    print("=== 🚀 FAST PROPERTY-BALANCED DATASET CREATION ===\n")
    
    # Load positive linkers
    try:
        positive_df = pd.read_csv('positive_linkers_combined.csv')
        positive_smiles = positive_df['canonical_smiles'].dropna().unique().tolist()
        print(f"📊 Loaded {len(positive_smiles)} positive linkers")
    except:
        print("❌ Error loading positive linkers. Using sample data.")
        # Fallback - you should replace this with your actual data loading
        return
    
    # Load decoy library
    decoy_library = load_large_decoy_library()
    
    # Create balanced dataset
    balanced_data = create_property_balanced_dataset(positive_smiles, decoy_library)
    
    # Convert to DataFrame
    balanced_df = pd.DataFrame(balanced_data)
    
    # Save dataset
    output_file = 'linkermind_fast_balanced_dataset.csv'
    balanced_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved balanced dataset to {output_file}")
    print(f"📦 Final dataset: {len(balanced_df)} molecules")
    print(f"   - {len(positive_smiles)} positives")
    print(f"   - {len(balanced_df) - len(positive_smiles)} negatives")
    
    # Analyze property balance
    print("\n=== 📈 PROPERTY BALANCE ANALYSIS ===")
    pos_props = [d for d in balanced_data if d['is_linker'] == 1]
    neg_props = [d for d in balanced_data if d['is_linker'] == 0]
    
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
        
        # Success criteria
        mw_diff = abs(pos_mw - neg_mw)
        if mw_diff < 100:
            print("✅ EXCELLENT: MW balance achieved!")
        elif mw_diff < 200:
            print("⚠️  ACCEPTABLE: MW difference reduced significantly")
        else:
            print("❌ NEEDS IMPROVEMENT: MW difference still too large")

if __name__ == "__main__":
    main()
