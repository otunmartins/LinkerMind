# debug_and_fix_dataset.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

def debug_dataset():
    print("=== 🔍 DEBUGGING DATASET ISSUES ===\n")
    
    # Load the dataset
    df = pd.read_csv('linkermind_final_balanced_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Class balance: {df['is_linker'].value_counts()}\n")
    
    # Check SMILES column
    smiles_col = 'smiles'
    print(f"SMILES column: '{smiles_col}'")
    print(f"First 5 SMILES:")
    for i, smi in enumerate(df[smiles_col].head()):
        print(f"  {i}: '{smi}'")
    
    # Check for NaN values
    print(f"\nNaN values in SMILES: {df[smiles_col].isna().sum()}")
    
    # Test SMILES parsing
    print("\n🔬 Testing SMILES parsing...")
    valid_count = 0
    invalid_smiles = []
    
    for idx, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            invalid_smiles.append((idx, smi))
        else:
            valid_count += 1
            
        if len(invalid_smiles) > 10:  # Stop after 10 invalid
            break
    
    print(f"Valid SMILES: {valid_count}/{len(df)}")
    if invalid_smiles:
        print(f"First 10 invalid SMILES:")
        for idx, smi in invalid_smiles[:10]:
            print(f"  Row {idx}: '{smi}'")

def create_working_dataset():
    """Create a working dataset with valid SMILES"""
    print("\n=== 🛠️ CREATING WORKING DATASET ===\n")
    
    # Load the original dataset
    df = pd.read_csv('linkermind_final_balanced_dataset.csv')
    
    # Filter valid SMILES
    valid_data = []
    invalid_count = 0
    
    for idx, row in df.iterrows():
        smi = str(row['smiles'])
        mol = Chem.MolFromSmiles(smi)
        
        if mol is not None:
            # Calculate basic properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            fraction_csp3 = Descriptors.FractionCsp3(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            
            valid_data.append({
                'smiles': smi,
                'is_linker': row['is_linker'],
                'mw': mw,
                'logp': logp,
                'tpsa': tpsa,
                'hbd': hbd,
                'hba': hba,
                'rotatable_bonds': rotatable_bonds,
                'aromatic_rings': aromatic_rings,
                'fraction_csp3': fraction_csp3,
                'heavy_atoms': heavy_atoms
            })
        else:
            invalid_count += 1
    
    # Create new dataset
    working_df = pd.DataFrame(valid_data)
    
    print(f"Original dataset: {len(df)} molecules")
    print(f"Valid molecules: {len(working_df)}")
    print(f"Invalid molecules: {invalid_count}")
    print(f"New dataset balance:")
    print(working_df['is_linker'].value_counts())
    
    # Save working dataset
    working_df.to_csv('linkermind_working_dataset.csv', index=False)
    print(f"\n💾 Saved working dataset to 'linkermind_working_dataset.csv'")
    
    return working_df

def quick_model_test(df):
    """Quick test to verify the dataset works"""
    print("\n=== 🧪 QUICK MODEL TEST ===\n")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    # Prepare features (exclude SMILES and target)
    feature_cols = [col for col in df.columns if col not in ['smiles', 'is_linker']]
    X = df[feature_cols].values
    y = df['is_linker'].values
    
    print(f"Feature matrix: {X.shape}")
    print(f"Features used: {feature_cols}")
    
    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Quick test AUC: {auc_score:.4f}")
    
    # Feature importance
    print("\n📊 Feature importances:")
    for name, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    # Step 1: Debug the current dataset
    debug_dataset()
    
    # Step 2: Create a working dataset
    working_df = create_working_dataset()
    
    # Step 3: Quick test to verify everything works
    quick_model_test(working_df)
    
    print("\n🎉 If the quick test shows a reasonable AUC (>0.5), then run the main training!")
    print("👉 Next command: python final_training_pipeline.py")
