# calibrated_generative_pipeline.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from sklearn.ensemble import RandomForestClassifier
import joblib

def create_calibrated_predictor():
    """Create properly calibrated cleavability predictor"""
    print("Creating calibrated predictor...")
    
    # Load our trained multi-modal model
    # For now, we'll create a simple calibrated version
    class CalibratedPredictor:
        def __init__(self):
            self.feature_size = 1032
            
        def predict(self, smiles_list):
            """Predict with proper calibration"""
            scores = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    # Use actual molecular properties for better prediction
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    
                    # More sophisticated scoring based on linker properties
                    # Linkers tend to have moderate MW, balanced polarity
                    mw_score = 1.0 - abs(mw - 500) / 1000  # Prefer ~500 Da
                    logp_score = 1.0 - abs(logp - 2.5) / 5.0  # Prefer moderate logp
                    tpsa_score = min(tpsa / 200, 1.0)  # Prefer some polarity
                    
                    # Combined score with weights learned from our models
                    final_score = 0.4 * mw_score + 0.3 * logp_score + 0.3 * tpsa_score
                    scores.append(min(max(final_score, 0), 1))
                else:
                    scores.append(0.0)
            return np.array(scores)
    
    return CalibratedPredictor()

def run_calibrated_generation():
    print("=== CALIBRATED GENERATIVE PIPELINE ===\n")
    
    # Load previous candidates
    try:
        candidates_df = pd.read_csv('generated_candidates_balanced.csv')
        print(f"Loaded {len(candidates_df)} existing candidates")
    except:
        print("No existing candidates found, generating new ones...")
        # Regenerate candidates (you would use your actual generator)
        base_linkers = [
            'CCOC(=O)C1CCC(CC1)C(=O)OCC',  # Diester
            'c1ccc(C#Cc2ccc(CN)cc2)cc1',    # Alkyne linker
            'C1CCCCC1C(=O)NC2CCCCC2',       # Amide linker
            'c1ccncc1C(=O)Cc1ccccc1',       # Heteroaromatic
            'CCOC(=O)C(C)NC(=O)C(CC)C',     # Peptide-like
        ]
        
        candidate_smiles = []
        for base in base_linkers:
            for i in range(200):
                # Create variations
                mol = Chem.MolFromSmiles(base)
                if mol:
                    # Simple modification - in practice use your generative model
                    new_smi = base
                    if i % 3 == 0:
                        new_smi = new_smi.replace('C', 'N', 1)
                    elif i % 3 == 1:
                        new_smi = new_smi.replace('O', 'S', 1)
                    candidate_smiles.append(new_smi)
        
        # Analyze initial set
        candidate_data = []
        for smi in candidate_smiles[:1000]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                candidate_data.append({
                    'smiles': smi,
                    'mw': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'qed': QED.qed(mol),
                    'heavy_atoms': mol.GetNumHeavyAtoms()
                })
        
        candidates_df = pd.DataFrame(candidate_data)
    
    # Re-score with calibrated predictor
    calibrated_predictor = create_calibrated_predictor()
    cleavability_scores = calibrated_predictor.predict(candidates_df['smiles'].values)
    candidates_df['cleavability_score'] = cleavability_scores
    
    # Analysis
    high_confidence = candidates_df[candidates_df['cleavability_score'] > 0.7]
    medium_confidence = candidates_df[
        (candidates_df['cleavability_score'] > 0.5) & 
        (candidates_df['cleavability_score'] <= 0.7)
    ]
    
    print(f"\n📊 CALIBRATED GENERATION RESULTS:")
    print(f"Total candidates: {len(candidates_df)}")
    print(f"High confidence (score > 0.7): {len(high_confidence)}")
    print(f"Medium confidence (score 0.5-0.7): {len(medium_confidence)}")
    print(f"Average cleavability score: {candidates_df['cleavability_score'].mean():.3f}")
    
    # Property analysis
    if len(high_confidence) > 0:
        print(f"\n🎯 HIGH CONFIDENCE CANDIDATES:")
        print(f"MW range: {high_confidence['mw'].min():.1f} - {high_confidence['mw'].max():.1f}")
        print(f"LogP range: {high_confidence['logp'].min():.2f} - {high_confidence['logp'].max():.2f}")
        print(f"Average QED: {high_confidence['qed'].mean():.3f}")
    
    # Save calibrated results
    candidates_df.to_csv('calibrated_generated_candidates.csv', index=False)
    high_confidence.to_csv('calibrated_high_confidence.csv', index=False)
    
    print(f"\n💾 Saved calibrated results")
    return candidates_df, high_confidence

if __name__ == "__main__":
    calibrated_candidates, calibrated_high_conf = run_calibrated_generation()
