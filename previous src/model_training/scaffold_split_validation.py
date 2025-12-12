# scaffold_split_validation.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

def scaffold_split_validation():
    print("=== SCAFFOLD-SPLIT VALIDATION (Final Rigor Test) ===\n")
    
    # Load the working dataset
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
    
    # Prepare features
    feature_cols = ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds', 'aromatic_rings', 'heavy_atoms', 'fraction_csp3']
    X = df_valid[feature_cols].values
    y = df_valid['is_linker'].values
    
    # Scaffold split
    from sklearn.model_selection import train_test_split
    
    unique_scaffolds = df_valid['scaffold'].unique()
    train_val_scaffolds, test_scaffolds = train_test_split(unique_scaffolds, test_size=0.2, random_state=42)
    
    train_val_mask = df_valid['scaffold'].isin(train_val_scaffolds)
    test_mask = df_valid['scaffold'].isin(test_scaffolds)
    
    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]
    X_test = X[test_mask]
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
    print(f"Test set linkers: {y_test.sum()}, decoys: {len(y_test)-y_test.sum()}")
    
    # Train Random Forest (best model)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_test_pred = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_pred)
    test_pr_auc = auc(test_recall, test_precision)
    
    print(f"\n🎯 SCAFFOLD-SPLIT RESULTS:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")
    
    # Compare with random split
    print(f"\n📊 COMPARISON WITH RANDOM SPLIT:")
    print(f"Scaffold-split AUC: {test_auc:.4f}")
    print(f"Random-split AUC:   0.8825")
    print(f"Performance drop:   {0.8825-test_auc:.4f}")
    
    if test_auc > 0.7:
        print("✅ EXCELLENT: Models generalize to novel scaffolds!")
    elif test_auc > 0.6:
        print("✅ GOOD: Reasonable generalization to novel scaffolds")
    else:
        print("⚠️  CAUTION: Limited generalization to novel scaffolds")

if __name__ == "__main__":
    scaffold_split_validation()
