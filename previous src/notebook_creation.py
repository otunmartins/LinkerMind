import json
import os

# Create the notebook content
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LinkerMind Milestone 1 Demo Notebook\n",
                "## ADC Linker Classification using Deep Learning\n",
                "\n",
                "This notebook demonstrates the working prototype of our AI system for classifying ADC linkers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from rdkit import Chem\n",
                "from rdkit.Chem import Draw, Descriptors\n",
                "import torch\n",
                "import pickle\n",
                "import sys\n",
                "import os\n",
                "\n",
                "# Add scripts directory to path\n",
                "sys.path.append('../scripts')\n",
                "from build_models import FeedForwardNetwork\n",
                "\n",
                "print(\"=== LinkerMind Milestone 1 Demo ===\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Loading the Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the curated dataset\n",
                "df = pd.read_csv('../data/linkermind_final_dataset_fixed.csv')\n",
                "print(f\"Dataset loaded: {len(df)} molecules\")\n",
                "print(f\"Linkers: {df['is_linker'].sum()}, Decoys: {len(df) - df['is_linker'].sum()}\")\n",
                "\n",
                "# Show data sources\n",
                "print(\"\\nData Sources:\")\n",
                "print(df['source'].value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Sample Molecules from Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display sample molecules\n",
                "linkers = df[df['is_linker'] == 1]['standard_smiles'].head(4)\n",
                "decoys = df[df['is_linker'] == 0]['standard_smiles'].head(4)\n",
                "\n",
                "linker_mols = [Chem.MolFromSmiles(s) for s in linkers]\n",
                "decoy_mols = [Chem.MolFromSmiles(s) for s in decoys]\n",
                "\n",
                "print(\"Sample Linkers:\")\n",
                "img_linkers = Draw.MolsToGridImage(linker_mols, molsPerRow=4, subImgSize=(200, 200), \n",
                "                                  legends=[f\"Linker {i+1}\" for i in range(len(linker_mols))])\n",
                "img_linkers"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Sample Decoys:\")\n",
                "img_decoys = Draw.MolsToGridImage(decoy_mols, molsPerRow=4, subImgSize=(200, 200),\n",
                "                                  legends=[f\"Decoy {i+1}\" for i in range(len(decoy_mols))])\n",
                "img_decoys"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Loading Trained Models"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set device\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f\"Using device: {device}\")\n",
                "\n",
                "# Load FFN model\n",
                "ffn_model = FeedForwardNetwork(input_size=2048)\n",
                "ffn_model.load_state_dict(torch.load('../models/ffn_model.pth', map_location=device))\n",
                "ffn_model.eval()\n",
                "print(\"✅ Feed-Forward Network loaded and ready for predictions\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Performance Summary"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load training results\n",
                "with open('../models/training_results.pkl', 'rb') as f:\n",
                "    results = pickle.load(f)\n",
                "\n",
                "ffn_metrics = results['ffn']['metrics']\n",
                "gnn_metrics = results['gnn']['metrics']\n",
                "\n",
                "performance_df = pd.DataFrame({\n",
                "    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],\n",
                "    'FFN': [\n",
                "        ffn_metrics['accuracy'],\n",
                "        ffn_metrics['precision'], \n",
                "        ffn_metrics['recall'],\n",
                "        ffn_metrics['f1'],\n",
                "        ffn_metrics['auc']\n",
                "    ],\n",
                "    'GNN': [\n",
                "        gnn_metrics['accuracy'],\n",
                "        gnn_metrics['precision'],\n",
                "        gnn_metrics['recall'], \n",
                "        gnn_metrics['f1'],\n",
                "        gnn_metrics['auc']\n",
                "    ]\n",
                "})\n",
                "\n",
                "print(\"Model Performance Comparison:\")\n",
                "print(performance_df.round(4))\n",
                "\n",
                "print(\"\\n🎯 Key Insight: FFN model achieved excellent performance with 97.48% AUC!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Interactive Prediction Demo"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def predict_linker(smiles):\n",
                "    \"\"\"Predict if a molecule is an ADC linker\"\"\"\n",
                "    from rdkit.Chem import AllChem\n",
                "    from rdkit import DataStructs\n",
                "    \n",
                "    # Validate SMILES\n",
                "    mol = Chem.MolFromSmiles(smiles)\n",
                "    if mol is None:\n",
                "        print(\"❌ Invalid SMILES string\")\n",
                "        return None\n",
                "    \n",
                "    # Generate Morgan fingerprint\n",
                "    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)\n",
                "    arr = np.zeros((1,))\n",
                "    DataStructs.ConvertToNumpyArray(fp, arr)\n",
                "    \n",
                "    # Predict using FFN model\n",
                "    with torch.no_grad():\n",
                "        input_tensor = torch.FloatTensor(arr).unsqueeze(0).to(device)\n",
                "        prediction = ffn_model(input_tensor).item()\n",
                "    \n",
                "    # Display results\n",
                "    print(f\"🔬 Analyzing: {smiles}\")\n",
                "    display(Draw.MolToImage(mol, size=(300, 300)))\n",
                "    \n",
                "    classification = \"LINKER\" if prediction > 0.5 else \"DECOY\"\n",
                "    confidence = max(prediction, 1-prediction)\n",
                "    \n",
                "    print(f\"📊 Prediction Score: {prediction:.4f}\")\n",
                "    print(f\"🏷️  Classification: {classification}\")\n",
                "    print(f\"🎯 Confidence: {confidence:.2%}\")\n",
                "    \n",
                "    if confidence > 0.9:\n",
                "        print(\"💪 High confidence prediction\")\n",
                "    elif confidence > 0.7:\n",
                "        print(\"👍 Good confidence prediction\")\n",
                "    else:\n",
                "        print(\"🤔 Lower confidence - molecule may be ambiguous\")\n",
                "    \n",
                "    return prediction\n",
                "\n",
                "print(\"✅ Prediction function ready! Use predict_linker('YOUR_SMILES') to test molecules.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Example Predictions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test example 1: Complex molecule (likely linker)\n",
                "print(\"Example 1: Complex functionalized molecule\")\n",
                "predict_linker(\"CCOC(=O)CNC(=O)C1CCCN1\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test example 2: Simple alkane (likely decoy)\n",
                "print(\"Example 2: Simple hydrocarbon\")\n",
                "predict_linker(\"CCCCCC\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test example 3: Cyclic molecule (likely decoy)\n",
                "print(\"Example 3: Simple cyclic compound\")\n",
                "predict_linker(\"C1CCCCC1\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Test Your Own Molecules"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# === TEST YOUR OWN MOLECULES HERE ===\n",
                "# Replace the SMILES string below with any molecule you want to test\n",
                "\n",
                "your_smiles = \"CCO\"  # ← Replace this with your SMILES string\n",
                "\n",
                "print(\"Testing your custom molecule:\")\n",
                "predict_linker(your_smiles)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebook with absolute path
notebook_path = "/mnt/f/Projects/Drug/LinkerMind/notebooks/LinkerMind_Milestone1_Demo.ipynb"

# Ensure the directory exists
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

# Write the notebook file
with open(notebook_path, 'w') as f:
    json.dump(notebook_content, f, indent=2)

print(f"✅ Jupyter notebook created at: {notebook_path}")
print("You can now open it with: jupyter notebook notebooks/LinkerMind_Milestone1_Demo.ipynb")
