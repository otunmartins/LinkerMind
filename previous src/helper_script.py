import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

plt.figure(figsize=(24, 5), dpi=300)
ax = plt.gca()
ax.axis('off')

def add_node(x, y, text):
    e = Ellipse((x, y), width=4.5, height=1, edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(e)
    ax.text(x, y, text, fontsize=10, ha='center', va='center')

def add_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', mutation_scale=15, linewidth=1.2)
    ax.add_patch(arrow)

# --- Node Placement (Keeps Your Exact Layout) ---

add_node(1, 2.5, "Dataset Collection\n(ChemBL, PubChem, Literature)")
add_node(5, 3.5, "Molecular Encoding\nRDKit, Graph Builder\nDescriptors, Fingerprints")
add_node(5, 1.5, "Mechanistic Features\nQM descriptors (XTB)\nMD-derived stability")

add_node(10, 3.5, "Mechanism-Aware GNN\n(Reactivity-aware attention)")
add_node(10, 1.5, "Transformer/Seq Model\n(Generative Linker Model)")

add_node(15, 2.5, "Multimodal Fusion Layer\nGNN + Mechanistic + Seq")

add_node(20, 3.5, "Property/Activity Prediction\nCleavage, Stability, Reactivity")
add_node(20, 1.5, "Optimized Linker Generation\nTransformer + RL/Scoring")

add_node(25, 2.5, "Final Candidates\nValidated, Ranked, Filtered")

# --- Arrows (Exact Original Directions) ---

add_arrow(3.3, 2.8, 3.9, 3.3)   # dataset → encoding
add_arrow(3.3, 2.2, 3.9, 1.7)   # dataset → mechanistic

add_arrow(7.3, 3.5, 8.7, 3.5)   # encoding → GNN
add_arrow(7.3, 1.5, 8.7, 1.5)   # mechanistic → transformer

add_arrow(11.3, 3.5, 13.7, 2.5)  # GNN → fusion
add_arrow(11.3, 1.5, 13.7, 2.5)  # transformer → fusion

add_arrow(16.3, 2.8, 18.7, 3.3)  # fusion → property
add_arrow(16.3, 2.2, 18.7, 1.7)  # fusion → optimized generation

add_arrow(21.3, 3.5, 23.7, 2.5)  # property → final
add_arrow(21.3, 1.5, 23.7, 2.5)  # optimized → final

plt.xlim(0, 30)
plt.ylim(0, 5)

plt.tight_layout()
plt.savefig("enhanced_architecture_diagram.png", dpi=300, bbox_inches='tight')
plt.show()
