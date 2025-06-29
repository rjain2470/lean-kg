
import pathlib, csv, random, torch, numpy as np
import networkx as nx
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

def write_edgelist_for_embeddings(
    G: nx.MultiDiGraph,
    out_dir: Union[str, pathlib.Path] = "kg_export",
    edges_name: str = "triples.tsv",      # now 3-column
    nodes_name: str = "nodes.csv",
    add_reverse: bool = False,
    drop_self_loops: bool = True,
    deduplicate: bool = True,
    relation_id: int = 0,                 # constant relation column
) -> None:
    """
    Exports the graph nodes and edges to CSV and TSV files for knowledge graph embedding.

    Args:
        G: The input NetworkX MultiDiGraph.
        out_dir: The directory to save the output files.
        edges_name: The name of the edges file (TSV format).
        nodes_name: The name of the nodes file (CSV format).
        add_reverse: Whether to add reverse edges to the triples.
        drop_self_loops: Whether to drop self-loop edges.
        deduplicate: Whether to deduplicate triples.
        relation_id: The constant ID to use for the relation column in the triples file.
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / nodes_name
    edges_path = out_dir / edges_name

    nodes = sorted(G.nodes())
    id_of  = {n: i for i, n in enumerate(nodes)}

    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name"])
        w.writerows((i, n) for n, i in id_of.items())

    # 2. write triples
    seen = set()
    written = 0
    with edges_path.open("w", encoding="utf-8") as f:
        for u, v in G.edges():
            if drop_self_loops and u == v:
                continue
            h, t = id_of[u], id_of[v]
            triple = (h, relation_id, t)
            if deduplicate and triple in seen:
                continue
            f.write(f"{h}	{relation_id}	{t}
")
            seen.add(triple)
            written += 1

            if add_reverse and h != t:
                triple_r = (t, relation_id, h)
                if not (deduplicate and triple_r in seen):
                    f.write(f"{t}	{relation_id}	{h}
")
                    seen.add(triple_r)
                    written += 1

    print(f"nodes.csv : {len(nodes):,} nodes  |  "
          f"triples.tsv : {written:,} triples")

def train_knowledge_graph_embedding(graph, model_name, emb_dim, epochs, batch_size, seed, device):
    """
    Trains a knowledge graph embedding model using PyKEEN.

    Args:
        graph: The input NetworkX graph.
        model_name: Name of the PyKEEN model to use.
        emb_dim: Embedding dimension.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        seed: Random seed.
        device: Device to use for training ('cuda' or 'cpu').

    Returns:
        PyKEEN PipelineResult object.
    """
    OUT = pathlib.Path("my_kg")
    OUT.mkdir(exist_ok=True)

    REL = "depends_on" # single relation label
    triples = [(u, REL, v) for u, v in graph.edges()] # u, v are Lean names

    ADD_REVERSE = False # optional: also write reverse edges
    if ADD_REVERSE:
        triples += [(t, REL, h) for h, _, t in triples]

    # shuffle & split 80 / 10 / 10
    random.seed(42); random.shuffle(triples)
    n = len(triples); n_tr = int(.8*n); n_va = int(.1*n)
    splits = dict(
        train = triples[:n_tr],
        valid = triples[n_tr:n_tr+n_va],
        test  = triples[n_tr+n_va:],
    )

    def write_tsv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="	").writerows(rows)

    for split, rows in splits.items():
        write_tsv(OUT / f"{split}.tsv", rows)

    print("Wrote", {k: len(v) for k, v in splits.items()}, "triples to", OUT)

    OUT_DIR = pathlib.Path("my_kg") # Define the directory where data files are located

    tf_train = TriplesFactory.from_path(OUT_DIR / "train.tsv")
    tf_valid = TriplesFactory.from_path(OUT_DIR / "valid.tsv")
    tf_test = TriplesFactory.from_path(OUT_DIR / "test.tsv")

    result = pipeline(
        model=model_name,
        training=tf_train,
        validation=tf_valid,
        testing=tf_test,
        model_kwargs=dict(embedding_dim=emb_dim),
        training_kwargs=dict(num_epochs=epochs, batch_size=batch_size),
        random_seed=seed,
        device=device,
    )
    return result

def summarize_pykeen_metrics(metric_dict: dict, section: str = "both") -> None:
    """
    Summarizes key evaluation metrics from a PyKEEN metric dictionary.

    Args:
        metric_dict: The dictionary containing evaluation metrics.
        section: The section of the metric dictionary to summarize ('head', 'tail', or 'both').

    Raises:
        KeyError: If the specified section is not in the metric dictionary.
    """
    if section not in metric_dict:
        raise KeyError(f"Section '{section}' not in metric dict.")

    block = metric_dict[section]["optimistic"]  # optimistic / realistic are same ≈

    mrr  = block["inverse_harmonic_mean_rank"]
    mr   = block["arithmetic_mean_rank"]
    hits = {k: block[f"hits_at_{k}"] for k in (1, 3, 10)}

    print(f" Test triples: {int(block['count']):,}")
    print(f"
=== Link-prediction quality ({section}) ===")
    print(f" MRR:        {mrr:8.4f}")
    print(f" MR:         {mr:8.2f}")
    for k, v in hits.items():
        print(f" Hits@{k:<2}:    {v:8.4f}")
    print("-" * 34)

# Note: score_triple, ent2id, rel2id, model, and device are not included here as they depend on the
# result of the training and are more for interactive use after training.
