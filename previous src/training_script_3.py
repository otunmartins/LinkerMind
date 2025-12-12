import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem

# Load your dataset
df = pd.read_csv('linkermind_final_dataset_fixed.csv')
print(f"Loaded dataset with {len(df)} molecules")

# Count cleavable motifs in the dataset
def count_cleavable_motifs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {}
    
    cleavable_patterns = {
        'Val-Cit Dipeptide': 'C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC(N)=O)C(=O)',
        'Disulfide Bond': 'S-S',
        'Hydrazone': 'C=NN',
        'Ester': 'C(=O)OC',
        'Carbamate': 'C(=O)NC',
        'Peptide Bond': 'C(=O)N',
    }
    
    motifs_found = {}
    for motif_name, smarts in cleavable_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            motifs_found[motif_name] = 1
    
    return motifs_found

# Count motifs across all linkers
print("Analyzing cleavable motifs in linker dataset...")
motif_counts = {}
total_linkers = 0

for idx, row in df.iterrows():
    if row['is_linker'] == 1:
        total_linkers += 1
        smiles = row['canonical_smiles']
        motifs = count_cleavable_motifs(smiles)
        for motif in motifs:
            motif_counts[motif] = motif_counts.get(motif, 0) + 1

print(f"Analyzed {total_linkers} linker molecules")

# Create a bar chart of motif frequencies
if motif_counts:
    # Sort by frequency
    sorted_motifs = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)
    motifs, counts = zip(*sorted_motifs)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(motifs, counts, color=['#2E86AB', '#A23B72', '#F18F01', '#73BFB8', '#C73E1D', '#3A7CA5'])
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Cleavable Motif Type', fontweight='bold')
    plt.ylabel('Number of Linkers', fontweight='bold')
    plt.title('Distribution of Cleavable Motifs in ADC Linker Dataset', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add total count annotation
    plt.text(0.02, 0.98, f'Total Linkers Analyzed: {total_linkers}', 
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('cleavable_motif_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('cleavable_motif_distribution.pdf', bbox_inches='tight')
    plt.show()
    
    print("\nCleavable Motif Distribution:")
    for motif, count in sorted_motifs:
        percentage = (count / total_linkers) * 100
        print(f"  {motif}: {count} linkers ({percentage:.1f}%)")

else:
    print("No cleavable motifs found in the dataset!")

print("\nAnalysis completed!")
