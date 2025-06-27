
# Lean-KG 🌪️

A Python library designed to generate and visualize dependency graphs of theorems formalized in Lean embedded in Euclidean N-space.

## About The Project 📈:
*insert info, i.e. "In particular, we parse the collection of roughly 600k Lean proofs for theorem names and dependencies, then create an embedding using PyKEEN."

## Getting Started 🚀

### Requirements 📝
* insert list of required libraries.

### Installation

You can install `lean-kg` using pip from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ lean-kg
```
## Usage 🤖

Here is a basic sample implementation of the package, assuming all relevant libraries are imported:
```python
import lean_kg
!pip install -q pykeen[all] # Install pykeen
!git clone https://github.com/leanprover-community/mathlib4.git # Clone mathlib4

# Example: Build a dependency graph
G_full = lean_kg.build_dependency_graph("mathlib4/Mathlib", num_workers=4) # Builds full dependency graph
G_sample = lean_kg.expand_sample_graph(G_pruned, k=1000, max_nodes=1000) # Samples a random subgraph of 1000 nodes
G_group = extract_subgraph_by_subdir(G_full, "GroupTheory") # Extracts subgraph of statements from Group Theory

# Example: Plot interactive graph on the Euclidean plane
fig = lean_kg.plot_euclidean_graph(G, color_by="weighted_edges")
fig.show()

# Example: Train a knowledge graph embedding (set hyperparameters beforehand).
result = train_knowledge_graph_embedding(
     G,
     MODEL_NAME, # example: BoxE, RotateE, ConvE...
     EMB_DIM,
     EPOCHS,
     BATCH_SIZE,
     SEED,
     'cuda' if torch.cuda.is_available() else 'cpu'
 )
lean_kg.summarize_pykeen_metrics(result.metric_results.to_dict()) # Outputs performance metrics of the embedding (MR, MRR, Hits@k)
```

## License
Distributed under the MIT License. See LICENSE.txt for more information.

## Contact
Ritik Jain - https://www.linkedin.com/in/ritik-jain-91a201220/ - rjain92682@gmail.com
