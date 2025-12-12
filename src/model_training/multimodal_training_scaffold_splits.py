# multimodal_training_scaffold_splits.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def generate_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Generate Morgan fingerprints for GNN-like features"""
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
        else:
            fingerprints.append(np.zeros(n_bits))
    return np.array(fingerprints)

def generate_mechanistic_descriptors(smiles_list):
    """Generate advanced mechanistic descriptors for cleavability"""
    descriptors = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            desc = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                mol.GetNumHeavyAtoms(),
                # Cleavage-related descriptors
                Descriptors.NumAmideBonds(mol),
                len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C;H3][O]'))),  # Methyl esters
                len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O'))),     # Carboxylic acids
                len(mol.GetSubstructMatches(Chem.MolFromSmarts('C#N'))),        # Nitriles
                len(mol.GetSubstructMatches(Chem.MolFromSmarts('S(=O)(=O)'))),  # Sulfones
            ]
            descriptors.append(desc)
        else:
            descriptors.append([0] * 13)
    return np.array(descriptors)

def create_multimodal_features(smiles_list):
    """Combine fingerprints + mechanistic descriptors"""
    fingerprints = generate_morgan_fingerprints(smiles_list)
    mechanistic = generate_mechanistic_descriptors(smiles_list)
    return np.concatenate([fingerprints, mechanistic], axis=1)

def scaffold_split_multimodal_training():
    print("=== MULTI-MODAL MODEL RE-TRAINING (Scaffold Splits) ===\n")
    
    # Load balanced dataset
    df = pd.read_csv('linkermind_working_dataset.csv')
    print(f"Dataset: {len(df)} molecules")
    
    # Generate scaffolds
    print("Generating molecular scaffolds...")
    scaffolds = []
    valid_indices = []
    
    for idx, smi in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)
            scaffolds.append(scaffold_smi)
            valid_indices.append(idx)
    
    df_valid = df.iloc[valid_indices].copy()
    df_valid['scaffold'] = scaffolds
    
    print(f"After scaffold generation: {len(df_valid)} molecules")
    print(f"Unique scaffolds: {df_valid['scaffold'].nunique()}")
    
    # Generate multi-modal features
    print("Generating multi-modal features (Fingerprints + Mechanistic)...")
    X_multi = create_multimodal_features(df_valid['smiles'].values)
    y = df_valid['is_linker'].values
    
    print(f"Multi-modal feature shape: {X_multi.shape}")
    
    # Scaffold split
    unique_scaffolds = df_valid['scaffold'].unique()
    train_val_scaffolds, test_scaffolds = train_test_split(unique_scaffolds, test_size=0.2, random_state=42)
    
    train_val_mask = df_valid['scaffold'].isin(train_val_scaffolds)
    test_mask = df_valid['scaffold'].isin(test_scaffolds)
    
    X_train_val = X_multi[train_val_mask]
    y_train_val = y[train_val_mask]
    X_test = X_multi[test_mask]
    y_test = y[test_mask]
    
    # Split train_val into train and validation
    train_val_df = df_valid[train_val_mask]
    train_val_scaffolds_unique = train_val_df['scaffold'].unique()
    train_scaffolds, val_scaffolds = train_test_split(train_val_scaffolds_unique, test_size=0.25, random_state=42)
    
    train_mask = train_val_df['scaffold'].isin(train_scaffolds)
    val_mask = train_val_df['scaffold'].isin(val_scaffolds)
    
    X_train = X_train_val[train_mask]
    y_train = y_train_val[train_mask]
    X_val = X_train_val[val_mask]
    y_val = y_train_val[val_mask]
    
    print(f"\nScaffold Split Results:")
    print(f"Train set: {len(X_train)} molecules")
    print(f"Validation set: {len(X_val)} molecules") 
    print(f"Test set: {len(X_test)} molecules")
    
    # Train Multi-Modal Model (Enhanced Random Forest)
    print("\n--- Training Multi-Modal Random Forest ---")
    multimodal_rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=30,
        min_samples_split=5,
        random_state=42, 
        n_jobs=-1
    )
    multimodal_rf.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = multimodal_rf.predict_proba(X_val)[:, 1]
    y_test_pred = multimodal_rf.predict_proba(X_test)[:, 1]
    
    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    val_precision, val_recall, _ = precision_recall_curve(y_val, y_val_pred)
    val_pr_auc = auc(val_recall, val_precision)
    
    test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_pred)
    test_pr_auc = auc(test_recall, test_precision)
    
    print(f"🎯 MULTI-MODAL MODEL RESULTS:")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Validation PR-AUC: {val_pr_auc:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")
    
    # Compare with baseline
    print(f"\n📊 COMPARISON WITH BASELINE:")
    print(f"Multi-Modal AUC: {test_auc:.4f}")
    print(f"Simple RF AUC:   0.8945")
    print(f"Improvement:     {test_auc-0.8945:+.4f}")
    
    return multimodal_rf, X_test, y_test, test_auc

if __name__ == "__main__":
    model, X_test, y_test, multimodal_auc = scaffold_split_multimodal_training()
