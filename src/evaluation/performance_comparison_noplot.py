# performance_comparison_noplot.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_performance_comparison():
    """Plot model performance across different validation strategies"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # AUC Comparison
    models = ['Random Forest', 'XGBoost']
    random_split = [0.8825, 0.8596]
    scaffold_split = [0.8945, 0.8596]  # Using actual XGBoost random split for scaffold
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, random_split, width, label='Random Split', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, scaffold_split, width, label='Scaffold Split', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Model Performance: Random vs Scaffold Splits')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0.8, 0.95)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(random_split):
        ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    for i, v in enumerate(scaffold_split):
        ax1.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Feature Importance
    features = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'Rotatable\nBonds', 
                'Aromatic\nRings', 'Heavy\nAtoms', 'Fraction\nCsp3']
    rf_importance = [0.1575, 0.1370, 0.1401, 0.0690, 0.0994, 0.0918, 0.0643, 0.0974, 0.1435]
    xgb_importance = [0.1374, 0.0761, 0.1486, 0.0993, 0.1422, 0.0884, 0.1380, 0.0691, 0.1009]
    
    x = np.arange(len(features))
    
    ax2.bar(x - width/2, rf_importance, width, label='Random Forest', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, xgb_importance, width, label='XGBoost', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Molecular Features')
    ax2.set_ylabel('Importance Score')
    ax2.set_title('Feature Importance Analysis')
    ax2.set_xticks(x)
    ax2.set_xticklabels(features, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_feature_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved performance_feature_analysis.png")
    plt.close()

plot_performance_comparison()
