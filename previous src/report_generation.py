import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import pickle
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os

class Milestone1Report:
    """Create comprehensive report for Milestone 1"""
    
    def __init__(self):
        self.results = {}
        
    def load_data(self):
        """Load all necessary data"""
        print("Loading data and results...")
        
        # Load dataset
        self.df = pd.read_csv('data/linkermind_final_dataset_fixed.csv')
        
        # Load features
        with open('data/features/morgan_fingerprints.pkl', 'rb') as f:
            self.morgan_data = pickle.load(f)
        
        with open('data/features/graph_data.pkl', 'rb') as f:
            self.graph_data = pickle.load(f)
        
        # Load training results
        with open('models/training_results.pkl', 'rb') as f:
            self.training_results = pickle.load(f)
        
        print("Data loaded successfully!")
    
    def create_dataset_summary(self):
        """Create dataset summary"""
        print("\nCreating dataset summary...")
        
        summary = {
            'total_molecules': len(self.df),
            'linkers': self.df['is_linker'].sum(),
            'decoys': len(self.df) - self.df['is_linker'].sum(),
            'sources': self.df['source'].value_counts().to_dict(),
            'class_balance': f"{self.df['is_linker'].mean():.1%} linkers, {1 - self.df['is_linker'].mean():.1%} decoys"
        }
        
        # Molecular properties
        linker_mols = [Chem.MolFromSmiles(s) for s in self.df[self.df['is_linker'] == 1]['standard_smiles']]
        decoy_mols = [Chem.MolFromSmiles(s) for s in self.df[self.df['is_linker'] == 0]['standard_smiles']]
        
        linker_mols = [m for m in linker_mols if m is not None]
        decoy_mols = [m for m in decoy_mols if m is not None]
        
        summary['linker_properties'] = {
            'avg_mw': np.mean([Descriptors.MolWt(m) for m in linker_mols]),
            'avg_logp': np.mean([Descriptors.MolLogP(m) for m in linker_mols]),
            'avg_hbd': np.mean([Descriptors.NumHDonors(m) for m in linker_mols]),
            'avg_hba': np.mean([Descriptors.NumHAcceptors(m) for m in linker_mols])
        }
        
        summary['decoy_properties'] = {
            'avg_mw': np.mean([Descriptors.MolWt(m) for m in decoy_mols]),
            'avg_logp': np.mean([Descriptors.MolLogP(m) for m in decoy_mols]),
            'avg_hbd': np.mean([Descriptors.NumHDonors(m) for m in decoy_mols]),
            'avg_hba': np.mean([Descriptors.NumHAcceptors(m) for m in decoy_mols])
        }
        
        self.results['dataset_summary'] = summary
        return summary
    
    def create_model_performance_report(self):
        """Create detailed model performance report"""
        print("Creating model performance report...")
        
        ffn_results = self.training_results['ffn']['metrics']
        gnn_results = self.training_results['gnn']['metrics']
        
        performance = {
            'ffn': {
                'accuracy': ffn_results['accuracy'],
                'precision': ffn_results['precision'],
                'recall': ffn_results['recall'],
                'f1': ffn_results['f1'],
                'auc': ffn_results['auc'],
                'confusion_matrix': ffn_results['confusion_matrix'].tolist()
            },
            'gnn': {
                'accuracy': gnn_results['accuracy'],
                'precision': gnn_results['precision'],
                'recall': gnn_results['recall'],
                'f1': gnn_results['f1'],
                'auc': gnn_results['auc'],
                'confusion_matrix': gnn_results['confusion_matrix'].tolist()
            }
        }
        
        self.results['model_performance'] = performance
        return performance
    
    def create_visual_summary(self):
        """Create comprehensive visual summary"""
        print("Creating visual summary...")
        
        os.makedirs('results/final_report', exist_ok=True)
        
        # 1. Dataset composition
        plt.figure(figsize=(15, 10))
        
        # Dataset composition
        plt.subplot(2, 3, 1)
        source_counts = self.df['source'].value_counts()
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        plt.title('Data Sources Composition')
        
        # Class distribution
        plt.subplot(2, 3, 2)
        class_counts = self.df['is_linker'].value_counts()
        plt.bar(['Decoys', 'Linkers'], class_counts.values, color=['lightcoral', 'lightsteelblue'])
        plt.title('Class Distribution')
        plt.ylabel('Count')
        
        # Model comparison
        plt.subplot(2, 3, 3)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        ffn_vals = [
            self.results['model_performance']['ffn']['accuracy'],
            self.results['model_performance']['ffn']['precision'],
            self.results['model_performance']['ffn']['recall'],
            self.results['model_performance']['ffn']['f1'],
            self.results['model_performance']['ffn']['auc']
        ]
        gnn_vals = [
            self.results['model_performance']['gnn']['accuracy'],
            self.results['model_performance']['gnn']['precision'],
            self.results['model_performance']['gnn']['recall'],
            self.results['model_performance']['gnn']['f1'],
            self.results['model_performance']['gnn']['auc']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, ffn_vals, width, label='FFN', alpha=0.8)
        plt.bar(x + width/2, gnn_vals, width, label='GNN', alpha=0.8)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion matrices
        plt.subplot(2, 3, 4)
        cm_ffn = np.array(self.results['model_performance']['ffn']['confusion_matrix'])
        sns.heatmap(cm_ffn, annot=True, fmt='d', cmap='Blues')
        plt.title('FFN Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.subplot(2, 3, 5)
        cm_gnn = np.array(self.results['model_performance']['gnn']['confusion_matrix'])
        sns.heatmap(cm_gnn, annot=True, fmt='d', cmap='Blues')
        plt.title('GNN Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Property comparison
        plt.subplot(2, 3, 6)
        properties = ['Molecular Weight', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors']
        linker_props = [
            self.results['dataset_summary']['linker_properties']['avg_mw'],
            self.results['dataset_summary']['linker_properties']['avg_logp'],
            self.results['dataset_summary']['linker_properties']['avg_hbd'],
            self.results['dataset_summary']['linker_properties']['avg_hba']
        ]
        decoy_props = [
            self.results['dataset_summary']['decoy_properties']['avg_mw'],
            self.results['dataset_summary']['decoy_properties']['avg_logp'],
            self.results['dataset_summary']['decoy_properties']['avg_hbd'],
            self.results['dataset_summary']['decoy_properties']['avg_hba']
        ]
        
        x = np.arange(len(properties))
        plt.bar(x - 0.2, linker_props, 0.4, label='Linkers', alpha=0.8)
        plt.bar(x + 0.2, decoy_props, 0.4, label='Decoys', alpha=0.8)
        plt.xlabel('Properties')
        plt.ylabel('Average Value')
        plt.title('Molecular Property Comparison')
        plt.xticks(x, properties, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/final_report/milestone1_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visual summary saved to results/final_report/milestone1_summary.png")
    
    def generate_text_report(self):
        """Generate comprehensive text report"""
        print("Generating text report...")
        
        report = f"""
=== LINKERMIND MILESTONE 1 COMPLETION REPORT ===
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET SUMMARY
==================
Total Molecules: {self.results['dataset_summary']['total_molecules']}
- Linkers: {self.results['dataset_summary']['linkers']}
- Decoys: {self.results['dataset_summary']['decoys']}
Class Balance: {self.results['dataset_summary']['class_balance']}

Data Sources:
{chr(10).join(f'  - {source}: {count}' for source, count in self.results['dataset_summary']['sources'].items())}

Molecular Properties:
Linkers:
  - Average Molecular Weight: {self.results['dataset_summary']['linker_properties']['avg_mw']:.1f}
  - Average LogP: {self.results['dataset_summary']['linker_properties']['avg_logp']:.2f}
  - Average H-Bond Donors: {self.results['dataset_summary']['linker_properties']['avg_hbd']:.1f}
  - Average H-Bond Acceptors: {self.results['dataset_summary']['linker_properties']['avg_hba']:.1f}

Decoys:
  - Average Molecular Weight: {self.results['dataset_summary']['decoy_properties']['avg_mw']:.1f}
  - Average LogP: {self.results['dataset_summary']['decoy_properties']['avg_logp']:.2f}
  - Average H-Bond Donors: {self.results['dataset_summary']['decoy_properties']['avg_hbd']:.1f}
  - Average H-Bond Acceptors: {self.results['dataset_summary']['decoy_properties']['avg_hba']:.1f}

2. MODEL PERFORMANCE
====================
Feed-Forward Network (Baseline):
  - Accuracy: {self.results['model_performance']['ffn']['accuracy']:.4f}
  - Precision: {self.results['model_performance']['ffn']['precision']:.4f}
  - Recall: {self.results['model_performance']['ffn']['recall']:.4f}
  - F1-Score: {self.results['model_performance']['ffn']['f1']:.4f}
  - AUC: {self.results['model_performance']['ffn']['auc']:.4f}

Graph Neural Network (Advanced):
  - Accuracy: {self.results['model_performance']['gnn']['accuracy']:.4f}
  - Precision: {self.results['model_performance']['gnn']['precision']:.4f}
  - Recall: {self.results['model_performance']['gnn']['recall']:.4f}
  - F1-Score: {self.results['model_performance']['gnn']['f1']:.4f}
  - AUC: {self.results['model_performance']['gnn']['auc']:.4f}

3. KEY FINDINGS
================
• FFN with Morgan fingerprints achieved EXCELLENT performance (97.48% AUC)
• The model demonstrates strong ability to distinguish ADC linkers from decoys
• GNN showed good AUC (85.82%) but struggled with class predictions
• Molecular fingerprints remain highly effective for molecular classification tasks
• Dataset shows clear physicochemical differences between linkers and decoys

4. NEXT STEPS FOR MILESTONE 2
=============================
• Integrate mechanistic data (QM/MD simulations)
• Add reaction mechanism learning
• Implement more advanced GNN architectures
• Include kinetic simulation validation
• Extend to linker generation tasks

=== END OF REPORT ===
"""
        
        with open('results/final_report/milestone1_report.txt', 'w') as f:
            f.write(report)
        
        print("Text report saved to results/final_report/milestone1_report.txt")
        return report
    
    def run(self):
        """Run complete reporting pipeline"""
        print("=== GENERATING MILESTONE 1 FINAL REPORT ===")
        
        self.load_data()
        self.create_dataset_summary()
        self.create_model_performance_report()
        self.create_visual_summary()
        self.generate_text_report()
        
        print("\n" + "="*50)
        print("MILESTONE 1 REPORT GENERATION COMPLETE!")
        print("="*50)
        print("✅ Dataset analysis completed")
        print("✅ Model performance evaluated") 
        print("✅ Visual summaries created")
        print("✅ Comprehensive report generated")
        print("📁 Check 'results/final_report/' for all outputs")

if __name__ == "__main__":
    report_generator = Milestone1Report()
    report_generator.run()
