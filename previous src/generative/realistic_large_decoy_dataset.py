# realistic_large_decoy_dataset.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import random
import os

def load_large_compound_libraries():
    """Load large molecules from available compound libraries"""
    print("📚 Loading large compound libraries for decoys...")
    
    large_compounds = []
    
    # Try to load from available datasets first
    available_files = [
        'chembl_extracted.csv',
        'pubchem_linkers.csv', 
        'literature_linkers.csv',
        'linkermind_final_dataset_fixed.csv'
    ]
    
    for file in available_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"📁 Loading compounds from {file}...")
                
                # Try different possible SMILES columns
                smiles_cols = [col for col in df.columns if 'smile' in col.lower() or 'canonical' in col.lower()]
                if smiles_cols:
                    smiles_col = smiles_cols[0]
                    for smi in df[smiles_col].dropna():
                        large_compounds.append(smi)
                    print(f"   Added {len(df)} molecules from {file}")
            except Exception as e:
                print(f"   Error reading {file}: {e}")
    
    # Add large synthetic compounds and common drug-like molecules
    large_synthetic = [
        # Large aromatic systems
        'c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43', # Large PAH
        'c1ccc2c(c1)c(c3ccc4c(c3)ccc(c4)c5ccc6c(c5)ccc(c6)c7ccc8c(c7)ccc(c8)c9ccc%10c(c9)ccc(c%10)c%11ccc%12c(c%11)ccc(c%12)c%13ccc%14c(c%13)ccc(c%14)c%15ccc%16c(c%15)ccc(c%16)c%17ccc%18c(c%17)ccc(c%18)c%19ccc%20c(c%19)ccc(c%20)c2)c2', # Very large
        'CCOC(=O)c1ccc(cc1)C(=O)OCC', # Diethyl terephthalate
        'CC(C)CC1CCC(CC1)C2CCC3C2(CCC4C3CCC5C4(C CC(C5)O)C)C', # Steroid-like
        'c1cc2c(cc1C(=O)O)oc1c(c2)ccc(c1)C(=O)O', # Large heterocyclic
        'CCOC(=O)C1CCC(CC1)C(=O)OCC', # Large aliphatic
        'c1ccc2c(c1)c(c3ccc4c(c3)ccc(c4)c5ccc6c(c5)ccc(c6)c2)c1ccc2c(c1)ccc(c2)c1ccc2c(c1)ccc(c2)c1ccc2c(c1)ccc(c2)c1', # Graphite-like
        # Polymer-like fragments
        'C(COC(=O)C1CCC(CC1)C(=O)O)COC(=O)C1CCC(CC1)C(=O)O',
        'C(CNC(=O)C1CCC(CC1)C(=O)N)CNC(=O)C1CCC(CC1)C(=O)N',
        'C(C(COC(=O)C1CCCCC1)OC(=O)C1CCCCC1)OC(=O)C1CCCCC1',
        # Large peptide-like
        'C(C(C(=O)O)N)C(C(C(C)C)(C)N)O', # Modified amino acids
        'C(CC(=O)O)C(C(=O)O)N', # Glutamate-like
        'C(C(CCCNC(=N)N)C(=O)O)N', # Arginine-like
    ]
    
    large_compounds.extend(large_synthetic)
    
    # Validate all compounds
    print("⚗️ Validating large compounds...")
    valid_compounds = []
    for smi in tqdm(large_compounds):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            if mw > 300:  # Only keep reasonably large molecules
                valid_compounds.append(smi)
    
    print(f"✅ Loaded {len(valid_compounds)} valid large compounds")
    return valid_compounds

def generate_large_decoys_by_combination():
    """Generate large decoys by combining smaller fragments"""
    print("🔗 Generating large decoys by combination...")
    
    # Start with smaller fragments
    base_fragments = [
        'c1ccccc1', 'C1CCCCC1', 'C1CCCC1', 'c1ccncc1', 'c1ccoc1', 'c1ccsc1',
        'CCO', 'CCN', 'CC(=O)O', 'CC(=O)N', 'CCOC(=O)C', 'CCNC(=O)C',
        'c1ccc(Cl)cc1', 'c1ccc(F)cc1', 'c1ccc(Br)cc1', 'c1ccc(I)cc1',
        'c1ccc(C#N)cc1', 'c1ccc(C=O)cc1', 'c1ccc(OC)cc1', 'c1ccc(CN)cc1',
        'C(C)(C)(C)', 'CC(C)C', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC'
    ]
    
    large_decoys = set()
    
    # Generate combinations (2-4 fragments)
    print("   Combining 2 fragments...")
    for i, frag1 in enumerate(base_fragments):
        for j, frag2 in enumerate(base_fragments):
            if i != j:
                # Different connection types
                combinations = [
                    f"{frag1}.{frag2}",  # Disconnected
                    f"{frag1}{frag2}",   # Connected
                    f"{frag1}C{frag2}",  # Connected with spacer
                    f"{frag1}CC{frag2}", # Connected with longer spacer
                    f"{frag1}C=C{frag2}", # Connected with double bond
                    f"{frag1}C#C{frag2}", # Connected with triple bond
                    f"{frag1}NC(=O){frag2}", # Amide connection
                    f"{frag1}OC(=O){frag2}", # Ester connection
                    f"{frag1}CC(=O)N{frag2}", # Reverse amide
                ]
                
                for combo in combinations:
                    mol = Chem.MolFromSmiles(combo)
                    if mol is not None:
                        mw = Descriptors.MolWt(mol)
                        if 300 <= mw <= 2500:  # Target MW range
                            large_decoys.add(combo)
                    
                    if len(large_decoys) > 5000:
                        break
            if len(large_decoys) > 5000:
                break
        if len(large_decoys) > 5000:
            break
    
    # Add some 3-fragment combinations for even larger molecules
    print("   Combining 3 fragments...")
    if len(large_decoys) < 10000:
        for i, frag1 in enumerate(base_fragments[:10]):
            for j, frag2 in enumerate(base_fragments[:10]):
                for k, frag3 in enumerate(base_fragments[:10]):
                    if i != j and i != k and j != k:
                        combo = f"{frag1}.{frag2}.{frag3}"
                        mol = Chem.MolFromSmiles(combo)
                        if mol is not None:
                            mw = Descriptors.MolWt(mol)
                            if mw > 400:
                                large_decoys.add(combo)
                        
                        if len(large_decoys) > 10000:
                            break
                if len(large_decoys) > 10000:
                    break
            if len(large_decoys) > 10000:
                break
    
    print(f"✅ Generated {len(large_decoys)} large decoy candidates")
    return list(large_decoys)

def create_realistic_balanced_dataset(positive_smiles):
    """Create balanced dataset using large decoys"""
    print("🎯 Creating realistic balanced dataset...")
    
    # Calculate properties for positives
    print("📊 Analyzing positive linkers...")
    positive_data = []
    for smi in tqdm(positive_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            props = {
                'smiles': smi,
                'is_linker': 1,
                'mw': Descriptors.MolWt(mol),
                'tpsa': Descriptors.TPSA(mol),
                'logp': Descriptors.MolLogP(mol)
            }
            positive_data.append(props)
    
    print(f"✅ Processed {len(positive_data)} positive linkers")
    
    # Generate large decoys
    large_decoys = generate_large_decoys_by_combination()
    
    # Calculate properties for decoys and filter by MW range
    print("📊 Analyzing and filtering decoys...")
    decoy_data = []
    pos_mw_values = [p['mw'] for p in positive_data]
    target_mw_min = np.percentile(pos_mw_values, 10)
    target_mw_max = np.percentile(pos_mw_values, 90)
    
    print(f"🎯 Target MW range: {target_mw_min:.1f} - {target_mw_max:.1f}")
    
    for smi in tqdm(large_decoys):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            if target_mw_min <= mw <= target_mw_max:
                props = {
                    'smiles': smi,
                    'is_linker': 0,
                    'mw': mw,
                    'tpsa': Descriptors.TPSA(mol),
                    'logp': Descriptors.MolLogP(mol)
                }
                decoy_data.append(props)
    
    print(f"📦 Found {len(decoy_data)} decoys in target MW range")
    
    # If still not enough, use all large decoys
    if len(decoy_data) < len(positive_data):
        print("⚠️  Using all large decoys (relaxed criteria)")
        decoy_data = []
        for smi in large_decoys:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                if mw > 300:  # At least somewhat large
                    props = {
                        'smiles': smi,
                        'is_linker': 0,
                        'mw': mw,
                        'tpsa': Descriptors.TPSA(mol),
                        'logp': Descriptors.MolLogP(mol)
                    }
                    decoy_data.append(props)
    
    # Sample the required number of decoys
    num_decoys_needed = min(len(positive_data), len(decoy_data))
    selected_decoys = random.sample(decoy_data, num_decoys_needed)
    
    # Combine datasets
    balanced_data = positive_data + selected_decoys
    random.shuffle(balanced_data)
    
    return balanced_data

def main():
    print("=== 🎯 REALISTIC LARGE-DECOYS BALANCED DATASET ===\n")
    
    # Load positive linkers
    try:
        positive_df = pd.read_csv('positive_linkers_combined.csv')
        positive_smiles = positive_df['canonical_smiles'].dropna().unique().tolist()
        print(f"📊 Loaded {len(positive_smiles)} positive linkers")
    except Exception as e:
        print(f"❌ Error loading positive linkers: {e}")
        return
    
    # Create balanced dataset with large decoys
    balanced_data = create_realistic_balanced_dataset(positive_smiles)
    
    # Convert to DataFrame and save
    balanced_df = pd.DataFrame(balanced_data)
    output_file = 'linkermind_realistic_balanced_dataset.csv'
    balanced_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved realistic balanced dataset to {output_file}")
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
        if mw_diff < 200:
            print("✅ EXCELLENT: Good MW balance achieved!")
            print("🎉 Ready for model training!")
        elif mw_diff < 400:
            print("⚠️  ACCEPTABLE: MW difference significantly reduced")
            print("🎉 Proceed with model training")
        else:
            print("❌ Still needs improvement, but much better than before")
            print("💡 Consider using this dataset for initial model testing")

if __name__ == "__main__":
    main()
