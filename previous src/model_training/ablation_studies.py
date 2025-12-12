# ablation_studies.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def run_ablation_studies():
    print("=== ABLATION STUDIES ===\n")
    
    # Load dataset
    df = pd.read_csv('linkermind_working_dataset.csv')
    
    # Generate different feature sets
    print("Generating feature sets for ablation...")
    
    # 1. Only Molecular Properties (no fingerprints)
    def get_property_features(smiles_list):
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                feat = [
                    Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
                    mol.GetNumHeavyAtoms()
                ]
                features.append(feat)
            else:
                features.append([0]*8)
        return np.array(features)
    
    # 2. Only Fingerprints (no properties)
    def get_fingerprint_features(smiles_list):
        fingerprints = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(1024))
        return np.array(fingerprints)
    
    # 3. Without Molecular Weight
    def get_features_no_mw(smiles_list):
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                feat = [
                    Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
                    mol.GetNumHeavyAtoms()
                ]
                features.append(feat)
            else:
                features.append([0]*7)
        return np.array(features)
    
    # 4. Without Mechanistic Descriptors (only fingerprints)
    def get_fingerprints_only(smiles_list):
        return get_fingerprint_features(smiles_list)
    
    # Generate all feature sets
    smiles_list = df['smiles'].values
    y = df['is_linker'].values
    
    X_properties = get_property_features(smiles_list)
    X_fingerprints = get_fingerprint_features(smiles_list) 
    X_no_mw = get_features_no_mw(smiles_list)
    X_fp_only = get_fingerprints_only(smiles_list)
    
    # Multi-modal (full features)
    X_multi = np.concatenate([X_fingerprints, X_properties], axis=1)
    
    feature_sets = {
        'Full Multi-Modal': X_multi,
        'Properties Only': X_properties,
        'Fingerprints Only': X_fingerprints,
        'No Molecular Weight': X_no_mw,
        'Fingerprints Only (No Properties)': X_fp_only
    }
    
    # Train and evaluate each feature set
    results = {}
    
    for name, X in feature_sets.items():
        print(f"\n--- Ablation: {name} ---")
        print(f"Feature shape: {X.shape}")
        
        # Simple train/test split for ablation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        
        results[name] = auc_score
        print(f"Test AUC: {auc_score:.4f}")
    
    # Results summary
    print("\n" + "="*50)
    print("📊 ABLATION STUDY RESULTS")
    print("="*50)
    
    full_performance = results['Full Multi-Modal']
    for name, auc in results.items():
        difference = auc - full_performance
        print(f"{name:<30}: {auc:.4f} ({difference:+.4f})")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"• Multi-Modal vs Properties Only: {results['Full Multi-Modal'] - results['Properties Only']:+.4f}")
    print(f"• Multi-Modal vs Fingerprints Only: {results['Full Multi-Modal'] - results['Fingerprints Only']:+.4f}")
    print(f"• Impact of Removing MW: {results['No Molecular Weight'] - results['Properties Only']:+.4f}")
    print(f"• Value of Mechanistic Features: {results['Full Multi-Modal'] - results['Fingerprints Only (No Properties)']:+.4f}")
    
    return results

if __name__ == "__main__":
    ablation_results = run_ablation_studies()
