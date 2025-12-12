import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import PandasTools
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_mol_properties(mol):
    """Calculate key molecular properties for matching"""
    if mol is None:
        return None
    
    return {
        'mw': Descriptors.MolWt(mol),
        'tpsa': CalcTPSA(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'logp': Descriptors.MolLogP(mol),
        'heavy_atoms': Descriptors.HeavyAtomCount(mol)
    }

def load_and_standardize_data():
    """Load your existing data"""
    print("Loading existing data...")
    
    # Load your positive linkers
    positive_df = pd.read_csv('positive_linkers_combined.csv')
    print(f"Loaded {len(positive_df)} positive linkers")
    
    # If you have the original dataset file
    try:
        full_df = pd.read_csv('linkermind_final_dataset_fixed.csv')
        negative_df = full_df[full_df['is_linker'] == 0].copy()
        print(f"Loaded {len(negative_df)} existing decoys")
    except:
        print("Creating new decoy matching process...")
        negative_df = None
    
    return positive_df, negative_df

def generate_zinc_decoys(positive_mols, num_decoys=804):
    """Generate decoys from ZINC-like properties"""
    print("Generating matched decoys from property distributions...")
    
    # Calculate property distributions from positive linkers
    positive_props = []
    for mol in positive_mols:
        if mol is not None:
            props = get_mol_properties(mol)
            if props:
                positive_props.append(props)
    
    prop_df = pd.DataFrame(positive_props)
    
    # Create decoys by sampling from property distributions
    decoys = []
    attempts = 0
    max_attempts = num_decoys * 10
    
    with tqdm(total=num_decoys, desc="Generating decoys") as pbar:
        while len(decoys) < num_decoys and attempts < max_attempts:
            attempts += 1
            
            # Sample properties from positive distribution
            sample_props = prop_df.sample(1).iloc[0]
            
            # Try to find or create molecules with similar properties
            # This is a simplified approach - in practice you'd use a real decoy database
            decoy_smiles = generate_simple_decoy(sample_props)
            
            if decoy_smiles:
                mol = Chem.MolFromSmiles(decoy_smiles)
                if mol:
                    decoy_props = get_mol_properties(mol)
                    if decoy_props:
                        # Check if properties are within reasonable range
                        if (abs(decoy_props['mw'] - sample_props['mw']) < 200 and
                            abs(decoy_props['tpsa'] - sample_props['tpsa']) < 50 and
                            abs(decoy_props['logp'] - sample_props['logp']) < 2):
                            decoys.append({
                                'smiles': decoy_smiles,
                                'is_linker': 0,
                                'source': 'generated_matched'
                            })
                            pbar.update(1)
    
    return pd.DataFrame(decoys)

def generate_simple_decoy(target_props):
    """Generate simple decoy molecules targeting specific properties"""
    # This is a simplified approach - in practice, use a real decoy database
    # or more sophisticated generation
    
    # Simple building blocks that can be combined
    scaffolds = [
        'C1CCCCC1',  # cyclohexane
        'c1ccccc1',  # benzene
        'C1CCNCC1',  # piperidine
        'C1COCCO1',  # dioxane
        'C1CCOCC1',  # morpholine-like
    ]
    
    side_chains = [
        'C', 'CC', 'CCC', 'C(C)(C)C', 'CO', 'CCO', 'CCCO',
        'CN', 'CCN', 'CCCN', 'C(=O)O', 'C(=O)N', 'C#N', 'CCl', 'CF'
    ]
    
    scaffold = random.choice(scaffolds)
    mol = Chem.MolFromSmiles(scaffold)
    
    # Add random side chains to adjust properties
    for _ in range(random.randint(1, 3)):
        if random.random() > 0.5:
            side_chain = random.choice(side_chains)
            # Simple substitution (this is very simplified)
            try:
                mol = Chem.ReplaceSubstructs(mol, 
                                           Chem.MolFromSmiles('C'), 
                                           Chem.MolFromSmiles(side_chain),
                                           replaceAll=True)[0]
            except:
                pass
    
    if mol:
        return Chem.MolToSmiles(mol)
    return None

def create_matched_dataset():
    """Main function to create matched decoy dataset"""
    print("=== Creating Matched Decoy Dataset ===")
    
    # Load existing data
    positive_df, negative_df = load_and_standardize_data()
    
    # Convert to molecules
    positive_mols = []
    for smiles in positive_df['smiles']:  # Adjust column name as needed
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            positive_mols.append(mol)
    
    print(f"Successfully processed {len(positive_mols)} positive linker molecules")
    
    # Generate matched decoys
    matched_decoys_df = generate_zinc_decoys(positive_mols, num_decoys=804)
    
    # Combine with positives
    positive_df['is_linker'] = 1
    positive_df['source'] = 'positive_linkers'
    
    # Ensure column consistency
    if 'smiles' not in positive_df.columns:
        # Try to find SMILES column
        for col in positive_df.columns:
            if 'smiles' in col.lower() or 'SMILES' in col:
                positive_df = positive_df.rename(columns={col: 'smiles'})
                break
    
    final_dataset = pd.concat([
        positive_df[['smiles', 'is_linker', 'source']],
        matched_decoys_df
    ], ignore_index=True)
    
    # Save the new balanced dataset
    final_dataset.to_csv('linkermind_matched_dataset.csv', index=False)
    print(f"Saved matched dataset with {len(final_dataset)} molecules")
    
    # Analyze property balance
    analyze_property_balance(final_dataset, positive_mols)
    
    return final_dataset

def analyze_property_balance(dataset, positive_mols):
    """Analyze if properties are now balanced"""
    print("\n=== Property Balance Analysis ===")
    
    positive_smiles = dataset[dataset['is_linker'] == 1]['smiles'].tolist()
    negative_smiles = dataset[dataset['is_linker'] == 0]['smiles'].tolist()
    
    pos_props = []
    neg_props = []
    
    for smiles in positive_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            pos_props.append(get_mol_properties(mol))
    
    for smiles in negative_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            neg_props.append(get_mol_properties(mol))
    
    pos_df = pd.DataFrame([p for p in pos_props if p])
    neg_df = pd.DataFrame([p for p in neg_props if p])
    
    print("Positive linkers (mean):")
    print(pos_df.mean())
    print("\nNegative decoys (mean):")
    print(neg_df.mean())
    print("\nDifference (positive - negative):")
    print(pos_df.mean() - neg_df.mean())

if __name__ == "__main__":
    create_matched_dataset()
