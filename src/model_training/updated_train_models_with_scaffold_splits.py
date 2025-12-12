# updated_train_models_with_scaffold_splits.py
"""
Updated Model Training Script for LinkerMind with Balanced Dataset and Scaffold Splits.
Includes XGBoost baseline.
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

# ----------------------------
# 1. Load the Balanced Dataset
# ----------------------------
print("Loading balanced dataset...")
df = pd.read_csv('linkermind_balanced_dataset.csv') # Update filename if different

# Assume the dataframe has columns: 'smiles', 'is_linker' (1 for linker, 0 for decoy), and other features
# If your feature columns are separate, adjust accordingly.
print(f"Dataset loaded with {len(df)} molecules.")
print(f"Linkers: {df['is_linker'].sum()}, Decoys: {len(df) - df['is_linker'].sum()}")

# ----------------------------
# 2. Generate Scaffolds for Splitting
# ----------------------------
print("Generating Bemis-Murcko scaffolds...")
def generate_scaffold(smiles):
    """Generates a Bemis-Murcko scaffold from a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    except:
        return None

df['scaffold'] = df['smiles'].apply(generate_scaffold)
# Drop molecules where scaffold generation failed
df = df.dropna(subset=['scaffold'])
print(f"Dataset size after scaffold generation: {len(df)}")

# ----------------------------
# 3. Create Scaffold-Based Train/Test/Validation Splits
# ----------------------------
print("Performing scaffold split...")

# First, split into a temporary holdout (80%) and a test set (20%)
# We do this at the scaffold level to ensure no scaffold leaks between train and test.
unique_scaffolds = df['scaffold'].unique()
train_val_scaffolds, test_scaffolds = train_test_split(unique_scaffolds, test_size=0.2, random_state=42)

# Get the molecules belonging to these scaffold sets
train_val_df = df[df['scaffold'].isin(train_val_scaffolds)]
test_df = df[df['scaffold'].isin(test_scaffolds)]

# Now, split the train_val set into train and validation, again by scaffold.
unique_train_val_scaffolds = train_val_df['scaffold'].unique()
train_scaffolds, val_scaffolds = train_test_split(unique_train_val_scaffolds, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.1

train_df = train_val_df[train_val_df['scaffold'].isin(train_scaffolds)]
val_df = train_val_df[train_val_df['scaffold'].isin(val_scaffolds)]

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Verify no scaffold overlap
train_scaffold_set = set(train_df['scaffold'])
val_scaffold_set = set(val_df['scaffold'])
test_scaffold_set = set(test_df['scaffold'])

assert train_scaffold_set.isdisjoint(val_scaffold_set), "Train and Val scaffolds overlap!"
assert train_scaffold_set.isdisjoint(test_scaffold_set), "Train and Test scaffolds overlap!"
assert val_scaffold_set.isdisjoint(test_scaffold_set), "Val and Test scaffolds overlap!"
print("✓ Scaffold split successful: No overlap between train, validation, and test sets.")

# ----------------------------
# 4. Prepare Features and Labels
# ----------------------------
# Identify feature columns. Adjust this logic based on your dataset.
# Let's assume all columns except 'smiles', 'is_linker', 'scaffold' are features.
feature_columns = [col for col in df.columns if col not in ['smiles', 'is_linker', 'scaffold']]
print(f"Using {len(feature_columns)} feature columns.")

X_train = train_df[feature_columns].values
y_train = train_df['is_linker'].values

X_val = val_df[feature_columns].values
y_val = val_df['is_linker'].values

X_test = test_df[feature_columns].values
y_test = test_df['is_linker'].values

print("Feature and label matrices prepared.")

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

# You can save the trained models for later use
# import joblib
# for model_name, res in results.items():
#     joblib.dump(res['model'], f'{model_name}_scaffold_split_model.joblib')

print("\nModel re-training with scaffold splits and XGBoost baseline is complete.")
