# fixed_train_models_scaffold_splits.py
"""
Fixed Model Training Script for LinkerMind with Proper Feature Handling
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def generate_basic_features(smiles):
    """Generate basic molecular features from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 10  # Return zeros if molecule is invalid
        
        from rdkit.Chem import Descriptors
        
        return [
            Descriptors.MolWt(mol),                    # Molecular Weight
            Descriptors.MolLogP(mol),                  # LogP
            Descriptors.TPSA(mol),                     # TPSA
            Descriptors.NumHDonors(mol),               # H-bond donors
            Descriptors.NumHAcceptors(mol),            # H-bond acceptors
            Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
            Descriptors.NumAromaticRings(mol),         # Aromatic rings
            Descriptors.FractionCsp3(mol),             # Fraction sp3 carbons
            mol.GetNumHeavyAtoms(),                    # Heavy atom count
            Descriptors.NumRadicalElectrons(mol)       # Radical electrons
        ]
    except:
        return [0] * 10

def main():
    print("=== FIXED Model Training with Scaffold Splits ===\n")
    
    # ----------------------------
    # 1. Load the Balanced Dataset
    # ----------------------------
    print("Loading balanced dataset...")
    try:
        # Try the new balanced dataset first
        df = pd.read_csv('linkermind_final_balanced_dataset.csv')
        print(f"✓ Loaded final balanced dataset: {len(df)} molecules")
    except:
        # Fallback to the original balanced dataset
        df = pd.read_csv('linkermind_balanced_dataset.csv')
        print(f"✓ Loaded original balanced dataset: {len(df)} molecules")
    
    print(f"Dataset loaded with {len(df)} molecules.")
    print(f"Linkers: {df['is_linker'].sum()}, Decoys: {len(df) - df['is_linker'].sum()}")
    
    # Show available columns
    print(f"\nAvailable columns: {list(df.columns)}")
    
    # ----------------------------
    # 2. Generate Features from SMILES
    # ----------------------------
    print("\nGenerating molecular features from SMILES...")
    
    # Find the SMILES column
    smiles_col = None
    for col in df.columns:
        if 'smile' in col.lower() or 'canonical' in col.lower() or col == 'smiles':
            smiles_col = col
            break
    
    if smiles_col is None:
        print("❌ ERROR: No SMILES column found!")
        print("Available columns:", list(df.columns))
        return
    
    print(f"✓ Using SMILES column: '{smiles_col}'")
    
    # Generate features for all molecules
    feature_data = []
    valid_indices = []
    
    for idx, smiles in enumerate(df[smiles_col]):
        features = generate_basic_features(str(smiles))
        if sum(features) > 0:  # Only keep valid molecules
            feature_data.append(features)
            valid_indices.append(idx)
    
    # Filter dataset to only valid molecules
    df_valid = df.iloc[valid_indices].copy()
    X = np.array(feature_data)
    y = df_valid['is_linker'].values
    
    print(f"✓ Generated features for {len(X)} valid molecules")
    print(f"Feature matrix shape: {X.shape}")
    
    # ----------------------------
    # 3. Generate Scaffolds for Splitting
    # ----------------------------
    print("\nGenerating Bemis-Murcko scaffolds...")
    
    def generate_scaffold(smiles):
        """Generates a Bemis-Murcko scaffold from a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            return scaffold_smiles
        except:
            return None
    
    df_valid['scaffold'] = df_valid[smiles_col].apply(generate_scaffold)
    # Drop molecules where scaffold generation failed
    df_valid = df_valid.dropna(subset=['scaffold'])
    print(f"Dataset size after scaffold generation: {len(df_valid)}")
    
    # Update X and y to match filtered dataset
    valid_indices_after_scaffold = df_valid.index
    X = X[[i for i, idx in enumerate(valid_indices) if idx in valid_indices_after_scaffold]]
    y = df_valid['is_linker'].values
    
    # ----------------------------
    # 4. Create Scaffold-Based Train/Test/Validation Splits
    # ----------------------------
    print("Performing scaffold split...")
    
    # First, split into a temporary holdout (80%) and a test set (20%)
    unique_scaffolds = df_valid['scaffold'].unique()
    train_val_scaffolds, test_scaffolds = train_test_split(unique_scaffolds, test_size=0.2, random_state=42)
    
    # Get the molecules belonging to these scaffold sets
    train_val_df = df_valid[df_valid['scaffold'].isin(train_val_scaffolds)]
    test_df = df_valid[df_valid['scaffold'].isin(test_scaffolds)]
    
    # Now, split the train_val set into train and validation, again by scaffold.
    unique_train_val_scaffolds = train_val_df['scaffold'].unique()
    train_scaffolds, val_scaffolds = train_test_split(unique_train_val_scaffolds, test_size=0.125, random_state=42)
    
    train_df = train_val_df[train_val_df['scaffold'].isin(train_scaffolds)]
    val_df = train_val_df[train_val_df['scaffold'].isin(val_scaffolds)]
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Get indices for splitting the feature matrix
    train_indices = train_df.index
    val_indices = val_df.index
    test_indices = test_df.index
    
    # Create feature matrices for each split
    X_train = X[[i for i, idx in enumerate(df_valid.index) if idx in train_indices]]
    y_train = y[[i for i, idx in enumerate(df_valid.index) if idx in train_indices]]
    
    X_val = X[[i for i, idx in enumerate(df_valid.index) if idx in val_indices]]
    y_val = y[[i for i, idx in enumerate(df_valid.index) if idx in val_indices]]
    
    X_test = X[[i for i, idx in enumerate(df_valid.index) if idx in test_indices]]
    y_test = y[[i for i, idx in enumerate(df_valid.index) if idx in test_indices]]
    
    # Verify no scaffold overlap
    train_scaffold_set = set(train_df['scaffold'])
    val_scaffold_set = set(val_df['scaffold'])
    test_scaffold_set = set(test_df['scaffold'])
    
    assert train_scaffold_set.isdisjoint(val_scaffold_set), "Train and Val scaffolds overlap!"
    assert train_scaffold_set.isdisjoint(test_scaffold_set), "Train and Test scaffolds overlap!"
    assert val_scaffold_set.isdisjoint(test_scaffold_set), "Val and Test scaffolds overlap!"
    print("✓ Scaffold split successful: No overlap between train, validation, and test sets.")
    
    # ----------------------------
    # 5. Train and Evaluate Models
    # ----------------------------
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on validation and test sets
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        # Calculate Precision-Recall AUC
        val_precision, val_recall, _ = precision_recall_curve(y_val, y_val_pred_proba)
        val_pr_auc = auc(val_recall, val_precision)
        
        test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
        test_pr_auc = auc(test_recall, test_precision)
        
        results[model_name] = {
            'model': model,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'val_pr_auc': val_pr_auc,
            'test_pr_auc': test_pr_auc
        }
        
        print(f"{model_name} Validation AUC: {val_auc:.4f}")
        print(f"{model_name} Test AUC: {test_auc:.4f}")
        print(f"{model_name} Validation PR AUC: {val_pr_auc:.4f}")
        print(f"{model_name} Test PR AUC: {test_pr_auc:.4f}")
    
    # ----------------------------
    # 6. Save the Results and Models (Optional)
    # ----------------------------
    print("\n--- Summary of Results on Test Set ---")
    for model_name, res in results.items():
        print(f"{model_name}: AUC = {res['test_auc']:.4f}, PR AUC = {res['test_pr_auc']:.4f}")
    
    # Feature importance analysis
    print("\n--- Feature Importance Analysis ---")
    feature_names = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 
                    'AromaticRings', 'FracSP3', 'HeavyAtoms', 'RadicalElectrons']
    
    rf_model = results['RandomForest']['model']
    feature_importances = rf_model.feature_importances_
    
    print("Random Forest Feature Importances:")
    for name, importance in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {importance:.4f}")
    
    print("\nModel training with scaffold splits completed successfully!")

if __name__ == "__main__":
    main()
