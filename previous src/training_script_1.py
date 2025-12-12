import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import seaborn as sns

# Simple version that generates fingerprints on the fly
df = pd.read_csv('linkermind_final_dataset_fixed.csv')
print(f"Dataset: {len(df)} molecules")
print(f"Linkers: {df['is_linker'].sum()}, Decoys: {len(df) - df['is_linker'].sum()}")

# Generate Morgan fingerprints
fingerprints = []
labels = []
valid_smiles = []

for idx, row in df.iterrows():
    smi = row['canonical_smiles']
    if pd.isna(smi):
        continue
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fingerprints.append(np.array(fp))
        labels.append(row['is_linker'])
        valid_smiles.append(smi)

fingerprint_array = np.array(fingerprints)
print(f"Generated {len(fingerprints)} valid fingerprints")

# Use PCA if t-SNE is too slow, or reduce dataset size
if len(fingerprints) > 1000:
    print("Large dataset detected. Using PCA for faster visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_embedded = pca.fit_transform(fingerprint_array)
    method = "PCA"
else:
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fingerprints)-1))
    X_embedded = tsne.fit_transform(fingerprint_array)
    method = "t-SNE"

# Create plot
plt.figure(figsize=(10, 8))

linker_mask = np.array(labels) == 1
decoy_mask = np.array(labels) == 0

plt.scatter(X_embedded[linker_mask, 0], X_embedded[linker_mask, 1], 
           c='#2E86AB', label='Linkers', alpha=0.7, s=50)
plt.scatter(X_embedded[decoy_mask, 0], X_embedded[decoy_mask, 1], 
           c='#A23B72', label='Decoys', alpha=0.7, s=50)

plt.xlabel(f'{method} Dimension 1', fontweight='bold')
plt.ylabel(f'{method} Dimension 2', fontweight='bold')
plt.title(f'Chemical Space: Linkers vs. Decoys ({method})', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chemical_space_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Visualization completed using {method}!")
