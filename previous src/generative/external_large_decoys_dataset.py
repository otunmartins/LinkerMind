# external_large_decoys_dataset.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import random
import os
import requests
import gzip
import io

def download_large_compound_library():
    """Download large compounds from public databases"""
    print("🌐 Downloading large compounds from public databases...")
    
    large_compounds = []
    
    # Try to download from DrugBank or other public sources
    public_sources = [
        # Common fragments and building blocks (larger ones)
        "CCOC(=O)C1=CC=CC=C1C(=O)OCC",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "C1=CC=C(C=C1)C2=CC=CC=C2C(=O)O",
        "C1=CC=C(C=C1)C2=CC=CC=C2C(=O)N",
        "CCOC(=O)C1CCC(CC1)C(=O)OCC",
        "C1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3",
        "C1=CC=C(C=C1)C2=CC=CC=C2C(=O)C3=CC=CC=C3",
        "C1=CC=C(C=C1)C2=CC=CC=C2N=C3C=CC=CC3=O",
        "C1=CC=C(C=C1)C2=CC=CC=C2C(=O)NC3=CC=CC=C3",
        "C1=CC=C(C=C1)C2=CC=CC=C2OC(=O)C3=CC=CC=C3",
        # Large steroid-like molecules
        "CC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        "CC12CCCC(C1CCC3C2CCC4(C3CCC4C(=O)O)C)O",
        # Large natural product-like
        "C1=CC(=CC=C1C2=CC(=C(C=C2)O)OC)O",
        "C1=CC(=CC=C1C2=C(C(=O)C3=C(C=CC=C3O2)O)O)O",
        "COC1=CC(=CC(=C1OC)OC)C2=CC(=O)C3=C(C=CC=C3O2)O",
        # Polymer fragments
        "C(COC(=O)C1=CC=CC=C1)COC(=O)C1=CC=CC=C1",
        "C(CNC(=O)C1=CC=CC=C1)CNC(=O)C1=CC=CC=C1",
        "C(C(COC(=O)C1=CC=CC=C1)OC(=O)C1=CC=CC=C1)OC(=O)C1=CC=CC=C1",
        # Large peptide-like
        "C(C(C(=O)NC(C)C(=O)O)NC(=O)C)NC(=O)C",
        "C(C(CC(=O)O)NC(=O)C(C)NC(=O)C)NC(=O)C",
        "C(C(CCCNC(=N)N)NC(=O)C(CC(=O)O)NC(=O)C)NC(=O)C",
    ]
    
    # Add some very large synthetic molecules
    very_large_molecules = [
        # Large aromatic systems
        "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43",  # Pyrene-like
        "c1ccc2c(c1)c(c3ccc4c(c3)ccc(c4)c5ccc6c(c5)ccc(c6)c2)c1",  # Very large PAH
        "C1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3C4=CC=CC=C4",  # Multiple aromatics
        "CCOC(=O)C1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)OCC",  # Large ester
        "C1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)C(=O)NC3=CC=CC=C3",  # Large amide
        "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=C(C=C3)C4=CC=CC=C4",  # Tetra-aryl
        "CCOC(=O)C1CCC(CC1)C(=O)C2CCC(CC2)C(=O)OCC",  # Large aliphatic
        "C1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)C3=CC=C(C=C3)C4=CC=CC=C4",  # Multi-aryl ketone
        "C1=CC=C(C=C1)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=CC=CC=C4",  # Large imine
        "C1=CC=C(C=C1)C2=CC=C(C=C2)OC(=O)C3=CC=C(C=C3)C4=CC=CC=C4",  # Large ester
    ]
    
    large_compounds.extend(public_sources)
    large_compounds.extend(very_large_molecules)
    
    # Validate and keep only large molecules
    print("⚗️ Validating and filtering large compounds...")
    valid_large_compounds = []
    for smi in tqdm(large_compounds):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            if mw > 400:  # Only keep truly large molecules
                valid_large_compounds.append(smi)
    
    print(f"✅ Found {len(valid_large_compounds)} valid large compounds")
    return valid_large_compounds

def generate_very_large_decoys():
    """Generate very large decoys by systematic combination"""
    print("🔗 Generating very large decoys...")
    
    # Core building blocks
    cores = [
        "c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
        "c1ccc(CN)cc1", "c1ccc(OC)cc1", "c1ccc(C=O)cc1", "c1ccc(C#N)cc1"
    ]
    
    # Linkers to connect cores
    linkers = [
        "C", "CC", "CCC", "C=C", "C#C", "C(=O)", "NC(=O)", "OC(=O)", 
        "C(=O)N", "C(=O)O", "CC(=O)N", "CC(=O)O", "N", "O", "S",
        "CCN", "CCO", "CCS", "CNC", "COC", "CSC"
    ]
    
    # End groups
    end_groups = [
        "Cl", "F", "Br", "I", "C", "CC", "CCC", "CN", "CO", "C(=O)O",
        "C(=O)N", "OC", "NC", "SC", "C#N", "C=O", "N", "O"
    ]
    
    large_decoys = set()
    
    print("   Building large molecules...")
    # Build molecules: core-linker-core-endgroup patterns
    for core1 in cores:
        for linker in linkers:
            for core2 in cores:
                for end_group in end_groups:
                    # Try different combinations
                    combinations = [
                        f"{core1}{linker}{core2}{end_group}",
                        f"{core1}.{core2}{linker}{end_group}", 
                        f"{core1}{linker}{core2}.{end_group}",
                        f"{core1}{linker}({core2}){end_group}",
                    ]
                    
                    for combo in combinations:
                        try:
                            mol = Chem.MolFromSmiles(combo)
                            if mol is not None:
                                mw = Descriptors.MolWt(mol)
                                if 500 <= mw <= 2500:  # Target very large range
                                    large_decoys.add(combo)
                        except:
                            pass
                        
                        if len(large_decoys) > 2000:
                            break
                    if len(large_decoys) > 2000:
                        break
                if len(large_decoys) > 2000:
                    break
            if len(large_decoys) > 2000:
                break
        if len(large_decoys) > 2000:
            break
    
    print(f"✅ Generated {len(large_decoys)} very large decoys")
    return list(large_decoys)

def create_final_balanced_dataset(positive_smiles):
    """Create the final balanced dataset using multiple strategies"""
    print("🎯 Creating final balanced dataset...")
    
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
    
    # Get decoys from multiple sources
    print("🔄 Collecting decoys from multiple sources...")
    
    # Source 1: Downloaded large compounds
    downloaded_decoys = download_large_compound_library()
    
    # Source 2: Generated very large decoys
    generated_decoys = generate_very_large_decoys()
    
    # Source 3: Use existing datasets that might contain large molecules
    existing_large_molecules = []
    for file in ['chembl_extracted.csv', 'pubchem_linkers.csv']:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                smiles_col = None
                for col in df.columns:
                    if 'smile' in col.lower() or 'canonical' in col.lower():
                        smiles_col = col
                        break
                
                if smiles_col:
                    for smi in df[smiles_col].dropna()[:500]:  # Sample first 500
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            mw = Descriptors.MolWt(mol)
                            if mw > 400:
                                existing_large_molecules.append(smi)
            except:
                pass
    
    # Combine all decoy sources
    all_decoys = list(set(downloaded_decoys + generated_decoys + existing_large_molecules))
    print(f"📦 Total decoy candidates: {len(all_decoys)}")
    
    # Calculate properties for decoys
    print("📊 Analyzing decoy properties...")
    decoy_data = []
    for smi in tqdm(all_decoys):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            props = {
                'smiles': smi,
                'is_linker': 0,
                'mw': mw,
                'tpsa': Descriptors.TPSA(mol),
                'logp': Descriptors.MolLogP(mol)
            }
            decoy_data.append(props)
    
    # Get MW distribution of positives
    pos_mw_values = [p['mw'] for p in positive_data]
    mw_25 = np.percentile(pos_mw_values, 25)
    mw_75 = np.percentile(pos_mw_values, 75)
    
    print(f"🎯 Target MW range (25th-75th percentile): {mw_25:.1f} - {mw_75:.1f}")
    
    # Filter decoys to match positive MW distribution
    filtered_decoys = [d for d in decoy_data if mw_25 <= d['mw'] <= mw_75]
    print(f"📊 Found {len(filtered_decoys)} decoys in target MW range")
    
    # If still not enough, use closest matches
    if len(filtered_decoys) < len(positive_data):
        print("⚠️  Using closest MW matches...")
        # Sort decoys by how close they are to median positive MW
        median_pos_mw = np.median(pos_mw_values)
        decoy_data.sort(key=lambda x: abs(x['mw'] - median_pos_mw))
        filtered_decoys = decoy_data[:len(positive_data)]
    
    # Sample the required number of decoys
    num_decoys_needed = min(len(positive_data), len(filtered_decoys))
    selected_decoys = random.sample(filtered_decoys, num_decoys_needed)
    
    # Combine datasets
    balanced_data = positive_data + selected_decoys
    random.shuffle(balanced_data)
    
    return balanced_data

def main():
    print("=== 🎯 FINAL BALANCED DATASET WITH LARGE DECOYS ===\n")
    
    # Load positive linkers
    try:
        positive_df = pd.read_csv('positive_linkers_combined.csv')
        positive_smiles = positive_df['canonical_smiles'].dropna().unique().tolist()
        print(f"📊 Loaded {len(positive_smiles)} positive linkers")
    except Exception as e:
        print(f"❌ Error loading positive linkers: {e}")
        return
    
    # Create final balanced dataset
    balanced_data = create_final_balanced_dataset(positive_smiles)
    
    # Convert to DataFrame and save
    balanced_df = pd.DataFrame(balanced_data)
    output_file = 'linkermind_final_balanced_dataset.csv'
    balanced_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved final balanced dataset to {output_file}")
    print(f"📦 Final dataset: {len(balanced_df)} molecules")
    print(f"   - {len([d for d in balanced_data if d['is_linker'] == 1])} positives")
    print(f"   - {len([d for d in balanced_data if d['is_linker'] == 0])} negatives")
    
    # Analyze property balance
    print("\n=== 📈 FINAL PROPERTY BALANCE ANALYSIS ===")
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
            print("🎉 EXCELLENT: Good MW balance achieved! Ready for model training!")
        elif mw_diff < 400:
            print("✅ GOOD: MW difference significantly reduced. Proceed with model training.")
        else:
            print("⚠️  MODERATE: MW difference still present but better.")
            print("💡 This dataset should work for initial model testing.")

if __name__ == "__main__":
    main()
