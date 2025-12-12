import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

# Load feature importance data
try:
    feature_imp = pd.read_csv('advanced_feature_importance.csv')
    print(f"Feature importance file loaded successfully!")
    print(f"Columns: {feature_imp.columns.tolist()}")
    print(f"Number of features: {len(feature_imp)}")
except FileNotFoundError:
    print("advanced_feature_importance.csv not found, trying feature_importance_scores.csv...")
    feature_imp = pd.read_csv('feature_importance_scores.csv')

# Check column names and adjust accordingly
print("\nFirst few rows of feature importance data:")
print(feature_imp.head())

# Identify the correct columns - common variations
importance_col = None
feature_col = None

for col in feature_imp.columns:
    if 'importance' in col.lower() or 'score' in col.lower():
        importance_col = col
    if 'feature' in col.lower() or 'descriptor' in col.lower():
        feature_col = col

if importance_col is None and len(feature_imp.columns) >= 2:
    # Assume first column is feature, second is importance
    feature_col = feature_imp.columns[0]
    importance_col = feature_imp.columns[1]

print(f"Using feature column: '{feature_col}', importance column: '{importance_col}'")

# Sort by importance and take top 15
feature_imp_sorted = feature_imp.sort_values(by=importance_col, ascending=True).tail(15)

# Create a more publication-ready plot
plt.figure(figsize=(12, 8))

# Color code by feature type with improved colors
colors = []
feature_categories = []

for feature in feature_imp_sorted[feature_col]:
    feature_lower = str(feature).lower()
    
    # QM/Electronic descriptors
    if any(x in feature_lower for x in ['homo', 'lumo', 'energy', 'charge', 'bde', 'gap', 'electro', 'dipole']):
        colors.append('#2E86AB')  # Blue
        feature_categories.append('QM/Electronic')
    
    # Mechanistic descriptors  
    elif any(x in feature_lower for x in ['cleav', 'motif', 'disulfide', 'peptide', 'ester', 'hydrazone', 'glucuronide', 'sulfhydryl']):
        colors.append('#A23B72')  # Purple
        feature_categories.append('Mechanistic')
    
    # Structural/MD descriptors
    elif any(x in feature_lower for x in ['sasa', 'rotatable', 'flexibility', 'torsion', 'shape', 'volume', 'surface']):
        colors.append('#F18F01')  # Orange
        feature_categories.append('Structural/MD')
    
    # Traditional descriptors
    elif any(x in feature_lower for x in ['molwt', 'logp', 'tpsa', 'hbd', 'hba', 'aromatic', 'ring']):
        colors.append('#73BFB8')  # Teal
        feature_categories.append('Traditional QSAR')
    
    # Other descriptors
    else:
        colors.append('#C73E1D')  # Red
        feature_categories.append('Other')

# Create the horizontal bar plot
y_pos = np.arange(len(feature_imp_sorted))
bars = plt.barh(y_pos, 
                feature_imp_sorted[importance_col], 
                color=colors, 
                alpha=0.8,
                height=0.7)

plt.yticks(y_pos, feature_imp_sorted[feature_col], fontsize=11)
plt.xlabel('Feature Importance Score', fontsize=13, fontweight='bold')
plt.title('Top 15 Most Important Features in Multi-Modal Model', 
          fontsize=15, fontweight='bold', pad=20)

# Add value labels on the bars
for i, (bar, value) in enumerate(zip(bars, feature_imp_sorted[importance_col])):
    plt.text(bar.get_width() + bar.get_width() * 0.01, 
             bar.get_y() + bar.get_height()/2, 
             f'{value:.3f}', 
             ha='left', va='center', 
             fontsize=10, fontweight='bold')

# Create custom legend for feature categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', label='QM/Electronic', alpha=0.8),
    Patch(facecolor='#A23B72', label='Mechanistic Motifs', alpha=0.8),
    Patch(facecolor='#F18F01', label='Structural/MD Proxies', alpha=0.8),
    Patch(facecolor='#73BFB8', label='Traditional QSAR', alpha=0.8),
    Patch(facecolor='#C73E1D', label='Other Descriptors', alpha=0.8)
]

plt.legend(handles=legend_elements, 
           loc='lower right', 
           framealpha=0.95,
           fontsize=10)

plt.grid(True, axis='x', alpha=0.3, linestyle='--')
plt.gca().set_axisbelow(True)

# Remove spines for cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('enhanced_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('enhanced_feature_importance.pdf', bbox_inches='tight', facecolor='white')
plt.show()

# Print summary statistics
print(f"\nFeature Importance Summary:")
print(f"Top feature: {feature_imp_sorted[feature_col].iloc[-1]} ({feature_imp_sorted[importance_col].iloc[-1]:.3f})")
print(f"Number of QM features in top 15: {feature_categories.count('QM/Electronic')}")
print(f"Number of Mechanistic features in top 15: {feature_categories.count('Mechanistic')}")
print(f"Number of Structural features in top 15: {feature_categories.count('Structural/MD')}")
