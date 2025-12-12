# Conceptual code
import pandas as pd

data_list = []
for result in all_results:
    # Check if the necessary data is available
    if 'molecule_structures' in result and result['molecule_structures']:
        structures = result['molecule_structures']
        # Prefer the canonical SMILES
        if 'canonical_smiles' in structures and structures['canonical_smiles']:
            smiles = structures['canonical_smiles']
            molecule_chembl_id = result['molecule_chembl_id']
            pref_name = result.get('pref_name', 'N/A')
            data_list.append({
                'molecule_chembl_id': molecule_chembl_id,
                'pref_name': pref_name,
                'canonical_smiles': smiles,
                'source': 'ChEMBL'
            })

# Create a DataFrame
chembl_df = pd.DataFrame(data_list)
print(f"Collected {len(chembl_df)} molecules with SMILES from ChEMBL.")
