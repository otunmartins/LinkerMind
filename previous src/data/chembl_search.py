from chembl_webresource_client.new_client import new_client
import pandas as pd
import pickle
import time

def search_chembl():
    """Search ChEMBL for ADC-related compounds with robust error handling"""
    molecule = new_client.molecule
    
    # Updated search terms - minimum 3 characters
    search_terms = [
        'Antibody-Drug Conjugate',
        'ADC linker', 
        'valine-citrulline',
        'val cit',  # alternative spacing
        'maleimidocaproyl',  # fixed spelling
        'malemidocaproyl',   # keep the original too
        'sulfo-spdb',
        'PEG linker',  # common linker component
        'cleavable linker',  # general category
        'biotin linker',  # related chemistry
        'disulfide linker',  # common linker type
        'peptide linker'  # broad category
    ]
    
    all_results = []
    
    for term in search_terms:
        print(f"Searching for: {term}")
        try:
            # Add delay to be respectful to the API
            time.sleep(0.5)
            
            results = molecule.search(term)
            if results:
                print(f"  Found {len(results)} results")
                all_results.extend(results)
            else:
                print(f"  No results found")
                
        except Exception as e:
            print(f"Error searching for {term}: {e}")
            continue
    
    print(f"Total preliminary results: {len(all_results)}")
    
    # Save results to a file for the next step
    with open('data/raw_chembl_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("Saved raw results to data/raw_chembl_results.pkl")
    return all_results

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    results = search_chembl()
