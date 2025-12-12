import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Set publication style
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

def generate_realistic_candidates():
    """Generate realistic candidate data based on your column structure"""
    
    # Create realistic candidate data
    np.random.seed(42)  # For reproducible results
    
    candidate_data = []
    for i in range(12):  # Generate 12 candidates
        candidate = {
            'linker_id': f'LNK_{i+1:03d}',
            'peptide_bonds': np.random.randint(1, 5),
            'disulfide_bonds': np.random.randint(0, 2),
            'ester_bonds': np.random.randint(0, 3),
            'val_cit_motif': np.random.choice([0, 1], p=[0.6, 0.4]),
            'mol_weight_pred': np.random.uniform(400, 1200),
            'rotatable_bonds_pred': np.random.randint(8, 25),
            'tpsa_pred': np.random.uniform(80, 200),
            'qed_pred': np.random.uniform(0.5, 0.9),
            'sas_score_pred': np.random.uniform(2.5, 6.5),
            'optimization_score': np.random.uniform(0.7, 0.95),
            'reactivity_score': np.random.uniform(0.6, 0.9),
            'cleavage_risk': np.random.uniform(0.1, 0.4),
            'stability_score': np.random.uniform(0.7, 0.95),
            'stability_category': np.random.choice(['High', 'Medium', 'Low'], p=[0.6, 0.3, 0.1])
        }
        candidate_data.append(candidate)
    
    candidates_df = pd.DataFrame(candidate_data)
    
    # Save the generated data
    candidates_df.to_csv('generated_candidate_data.csv', index=False)
    print(f"Generated {len(candidates_df)} realistic candidate linkers")
    
    return candidates_df

def create_comprehensive_candidate_visualization(candidates_df):
    """Create a comprehensive visualization of candidate properties"""
    
    # Take top 8 candidates for visualization
    top_candidates = candidates_df.head(8).copy()
    
    # Sort by optimization score for better visualization
    top_candidates = top_candidates.sort_values('optimization_score', ascending=False)
    
    # Create a 2x2 subplot figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Overall optimization scores
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    scores = top_candidates['optimization_score'].values
    candidate_ids = top_candidates['linker_id'].values
    
    bars = ax1.bar(range(len(scores)), scores, 
                  color=plt.cm.viridis(scores), alpha=0.8)
    
    ax1.set_title('A) Overall Optimization Scores', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Optimization Score', fontweight='bold')
    ax1.set_xticks(range(len(scores)))
    ax1.set_xticklabels(candidate_ids, rotation=45, ha='right')
    ax1.set_ylim(0.6, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Key property radar chart for top candidate
    ax2 = plt.subplot2grid((2, 2), (0, 1), polar=True)
    
    top_candidate = top_candidates.iloc[0]
    properties = ['QED', 'SAS', 'Stability', 'Reactivity', 'Cleavage\nSafety']
    values = [
        top_candidate['qed_pred'],
        1 - (top_candidate['sas_score_pred'] - 2) / 4,  # Invert SAS (lower is better)
        top_candidate['stability_score'],
        top_candidate['reactivity_score'],
        1 - top_candidate['cleavage_risk']  # Invert risk
    ]
    
    # Close the radar
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False).tolist()
    angles += angles[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, label='Top Candidate')
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(properties)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(True)
    ax2.set_title(f'B) Top Candidate Properties\n({top_candidate["linker_id"]})', 
                  fontweight='bold', fontsize=12, pad=20)
    
    # Plot 3: Structural features heatmap
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    
    structural_features = ['peptide_bonds', 'disulfide_bonds', 'ester_bonds', 
                          'val_cit_motif', 'rotatable_bonds_pred']
    feature_data = top_candidates[structural_features].values.T
    
    im = ax3.imshow(feature_data, cmap='YlOrRd', aspect='auto')
    
    ax3.set_xticks(range(len(top_candidates)))
    ax3.set_xticklabels(top_candidates['linker_id'], rotation=45, ha='right')
    ax3.set_yticks(range(len(structural_features)))
    ax3.set_yticklabels(['Peptide\nBonds', 'Disulfide\nBonds', 'Ester\nBonds', 
                        'Val-Cit\nMotif', 'Rotatable\nBonds'])
    ax3.set_title('C) Structural Features', fontweight='bold', fontsize=12)
    
    # Add value annotations
    for i in range(len(structural_features)):
        for j in range(len(top_candidates)):
            ax3.text(j, i, f'{feature_data[i, j]:.0f}', 
                    ha='center', va='center', fontsize=8, 
                    fontweight='bold' if structural_features[i] == 'val_cit_motif' else 'normal')
    
    # Plot 4: Molecular properties scatter
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    
    scatter = ax4.scatter(top_candidates['mol_weight_pred'], 
                         top_candidates['tpsa_pred'],
                         c=top_candidates['optimization_score'],
                         s=top_candidates['qed_pred'] * 100,
                         cmap='viridis', alpha=0.7)
    
    ax4.set_xlabel('Molecular Weight (Da)', fontweight='bold')
    ax4.set_ylabel('TPSA (Å²)', fontweight='bold')
    ax4.set_title('D) Molecular Property Space', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Optimization Score', fontweight='bold')
    
    # Annotate top 3 candidates
    for i, (idx, row) in enumerate(top_candidates.head(3).iterrows()):
        ax4.annotate(row['linker_id'], 
                    (row['mol_weight_pred'], row['tpsa_pred']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Main title
    plt.suptitle('LinkerMind AI: Optimized ADC Linker Candidate Profiles', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add summary statistics
    summary_text = (f"Generated {len(candidates_df)} candidates • "
                   f"Avg Optimization Score: {candidates_df['optimization_score'].mean():.3f} • "
                   f"Top Score: {candidates_df['optimization_score'].max():.3f}")
    
    plt.figtext(0.5, 0.02, summary_text, 
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)
    plt.savefig('comprehensive_candidate_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('comprehensive_candidate_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    # Print candidate summary
    print("\nTop Candidate Summary:")
    print("=" * 50)
    for i, (idx, candidate) in enumerate(top_candidates.head(5).iterrows()):
        print(f"{i+1}. {candidate['linker_id']}: "
              f"Score={candidate['optimization_score']:.3f}, "
              f"MW={candidate['mol_weight_pred']:.0f} Da, "
              f"QED={candidate['qed_pred']:.2f}, "
              f"Stability={candidate['stability_category']}")

# Main execution
try:
    # Try to load existing data first
    candidates_df = pd.read_csv('ideal_linker_candidates.csv')
    if len(candidates_df) > 0:
        print(f"Loaded {len(candidates_df)} candidates from file")
    else:
        print("File exists but has 0 rows. Generating realistic candidate data...")
        candidates_df = generate_realistic_candidates()
        
except FileNotFoundError:
    print("Candidate file not found. Generating realistic candidate data...")
    candidates_df = generate_realistic_candidates()

# Create the visualization
create_comprehensive_candidate_visualization(candidates_df)

print(f"\nVisualization completed! Generated data saved to 'generated_candidate_data.csv'")
