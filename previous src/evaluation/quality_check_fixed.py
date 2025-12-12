import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def quality_check():
    """Perform comprehensive quality checks on the fixed dataset"""
    
    # Load fixed dataset
    df = pd.read_csv('data/linkermind_final_dataset_fixed.csv')
    
    print("=== DATASET QUALITY CHECK ===")
    print(f"Total molecules: {len(df)}")
    print(f"Positive linkers: {len(df[df['is_linker'] == 1])}")
    print(f"Negative decoys: {len(df[df['is_linker'] == 0])}")
    
    # Check data sources
    print("\n=== DATA SOURCES ===")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    # Calculate molecular properties
    print("\n=== MOLECULAR PROPERTY ANALYSIS ===")
    
    properties = []
    valid_molecules = []
    
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['standard_smiles'])
        if mol is not None:
            props = {
                'is_linker': row['is_linker'],
                'source': row['source'],
                'smiles': row['standard_smiles'],
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'ring_count': Descriptors.RingCount(mol),
                'tpsa': Descriptors.TPSA(mol)
            }
            properties.append(props)
            valid_molecules.append(mol)
    
    props_df = pd.DataFrame(properties)
    
    # Print property statistics by class
    print("\n--- PROPERTY STATISTICS BY CLASS ---")
    for label in [0, 1]:
        class_data = props_df[props_df['is_linker'] == label]
        class_name = "DECOYS" if label == 0 else "LINKERS"
        print(f"\n{class_name} (n={len(class_data)}):")
        
        stats = {
            'Molecular Weight': f"{class_data['mw'].mean():.1f} ± {class_data['mw'].std():.1f}",
            'LogP': f"{class_data['logp'].mean():.2f} ± {class_data['logp'].std():.2f}",
            'H-Bond Donors': f"{class_data['hbd'].mean():.1f} ± {class_data['hbd'].std():.1f}",
            'H-Bond Acceptors': f"{class_data['hba'].mean():.1f} ± {class_data['hba'].std():.1f}",
            'Rotatable Bonds': f"{class_data['rotatable_bonds'].mean():.1f} ± {class_data['rotatable_bonds'].std():.1f}",
            'Heavy Atoms': f"{class_data['heavy_atoms'].mean():.1f} ± {class_data['heavy_atoms'].std():.1f}",
            'Ring Count': f"{class_data['ring_count'].mean():.1f} ± {class_data['ring_count'].std():.1f}",
            'TPSA': f"{class_data['tpsa'].mean():.1f} ± {class_data['tpsa'].std():.1f}"
        }
        
        for prop, value in stats.items():
            print(f"  {prop}: {value}")
    
    # Create visualization
    create_property_plots(props_df)
    
    # Draw sample molecules
    draw_sample_molecules(props_df, valid_molecules)

def create_property_plots(props_df):
    """Create plots comparing linker vs decoy properties"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    properties = ['mw', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'heavy_atoms', 'ring_count', 'tpsa']
    titles = ['Molecular Weight', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors', 
              'Rotatable Bonds', 'Heavy Atoms', 'Ring Count', 'TPSA']
    
    colors = ['lightcoral', 'lightsteelblue']
    
    for i, (prop, title) in enumerate(zip(properties, titles)):
        if i >= len(axes):
            break
            
        # Prepare data for box plot
        linker_data = props_df[props_df['is_linker'] == 1][prop].dropna()
        decoy_data = props_df[props_df['is_linker'] == 0][prop].dropna()
        
        data_to_plot = [decoy_data, linker_data]
        
        # Create box plot
        bp = axes[i].boxplot(data_to_plot, labels=['Decoys', 'Linkers'], 
                           patch_artist=True, showfliers=False)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[i].set_title(f'Distribution of {title}', fontweight='bold')
        axes[i].set_ylabel(title)
        axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(properties) < len(axes):
        axes[len(properties)].set_visible(False)
    
    plt.suptitle('Molecular Property Distributions: Linkers vs Decoys', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/property_distributions.png', dpi=300, bbox_inches='tight')
    print("\nSaved property distributions to data/property_distributions.png")

def draw_sample_molecules(props_df, valid_molecules):
    """Draw sample molecules from each class"""
    
    # Get samples from each class
    linker_samples = props_df[props_df['is_linker'] == 1].head(8)
    decoy_samples = props_df[props_df['is_linker'] == 0].head(8)
    
    # Get corresponding molecules
    linker_mols = []
    decoy_mols = []
    
    for _, row in linker_samples.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            linker_mols.append(mol)
    
    for _, row in decoy_samples.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            decoy_mols.append(mol)
    
    # Draw molecules
    if linker_mols:
        img = Draw.MolsToGridImage(linker_mols, molsPerRow=4, subImgSize=(200, 200),
                                 legends=[f"Linker {i+1}" for i in range(len(linker_mols))])
        img.save('data/sample_linkers.png')
        print("Saved sample linkers to data/sample_linkers.png")
    
    if decoy_mols:
        img = Draw.MolsToGridImage(decoy_mols, molsPerRow=4, subImgSize=(200, 200),
                                 legends=[f"Decoy {i+1}" for i in range(len(decoy_mols))])
        img.save('data/sample_decoys.png')
        print("Saved sample decoys to data/sample_decoys.png")

def calculate_diversity(props_df):
    """Calculate chemical diversity of the dataset"""
    
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    import numpy as np
    
    print("\n=== CHEMICAL DIVERSITY ANALYSIS ===")
    
    # Generate fingerprints for diversity calculation
    fingerprints = []
    valid_smiles = []
    
    for _, row in props_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp)
            valid_smiles.append(row['smiles'])
    
    if len(fingerprints) < 2:
        print("Not enough valid molecules for diversity calculation")
        return
    
    # Calculate pairwise Tanimoto similarities
    similarity_matrix = []
    for i in range(len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        similarity_matrix.extend(similarities)
    
    avg_similarity = np.mean(similarity_matrix) if similarity_matrix else 0
    print(f"Average pairwise Tanimoto similarity: {avg_similarity:.3f}")
    
    # Calculate diversity by class
    linker_fps = [fp for fp, label in zip(fingerprints, props_df['is_linker']) if label == 1]
    decoy_fps = [fp for fp, label in zip(fingerprints, props_df['is_linker']) if label == 0]
    
    if len(linker_fps) > 1:
        linker_similarities = []
        for i in range(len(linker_fps)):
            similarities = DataStructs.BulkTanimotoSimilarity(linker_fps[i], linker_fps[:i])
            linker_similarities.extend(similarities)
        avg_linker_sim = np.mean(linker_similarities) if linker_similarities else 0
        print(f"Linker diversity (avg similarity): {avg_linker_sim:.3f}")
    
    if len(decoy_fps) > 1:
        decoy_similarities = []
        for i in range(len(decoy_fps)):
            similarities = DataStructs.BulkTanimotoSimilarity(decoy_fps[i], decoy_fps[:i])
            decoy_similarities.extend(similarities)
        avg_decoy_sim = np.mean(decoy_similarities) if decoy_similarities else 0
        print(f"Decoy diversity (avg similarity): {avg_decoy_sim:.3f}")

if __name__ == "__main__":
    quality_check()
    # Load data for diversity calculation
    df = pd.read_csv('data/linkermind_final_dataset_fixed.csv')
    props_df = calculate_diversity_prep(df)
    calculate_diversity(props_df)
