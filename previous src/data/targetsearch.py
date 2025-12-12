# This is a conceptual outline for your notebook.
from chembl_webresource_client.new_client import new_client

# Initialize the client
molecule = new_client.molecule

# Search for molecules with "ADC" or "Antibody-Drug Conjugate" in their description
# We will also search for specific, well-known linker chemistries.
search_terms = ['Antibody-Drug Conjugate', 'ADC linker', 'valine-citrulline', 'vc', 'mc', 'malemidocaproyl', 'sulfo-spdb']
all_results = []

for term in search_terms:
    print(f"Searching for: {term}")
    try:
        results = molecule.search(term)  # This searches various fields including molecule synonyms and descriptions
        all_results.extend(results)
    except Exception as e:
        print(f"Error searching for {term}: {e}")

print(f"Total preliminary results: {len(all_results)}")
