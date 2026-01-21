import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import connected_components

import requests
import zipfile
import io
import os

DATA_URL = "https://networksciencebook.com/translations/en/resources/networks.zip"
DATA_DIR = "data"

def download_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)

    response = requests.get(DATA_URL)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(DATA_DIR)

    print("Datasets downloaded and extracted.")


def load_edge_list(file_path):
    edges = np.loadtxt(file_path, dtype=int)
    return edges

def create_simple_undirected_adjacency(edges):
    # convert to 0-based indexing if needed
    if edges.min() == 1:
        edges = edges - 1

    # remove self-loops
    edges = edges[edges[:, 0] != edges[:, 1]]

    # make undirected by sorting endpoints
    edges = np.sort(edges, axis=1)

    # remove duplicate edges
    edges = np.unique(edges, axis=0)

    num_nodes = edges.max() + 1

    A = csc_matrix(
    (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
    shape=(num_nodes, num_nodes)
    )

    A = A + A.T

    return A


def compute_graph_metrics(adj_matrix):
    # Compute connected components
    num_components, labels = connected_components(adj_matrix, directed=False)

    # Size of each component
    component_sizes = np.bincount(labels)
    largest_component_size = component_sizes.max()

    # Node and edge counts
    num_nodes = adj_matrix.shape[0]
    num_edges = adj_matrix.nnz // 2# Each edge is counted twice in undirected graphs

    return {
        "Number of Nodes": int(num_nodes),
        "Number of Edges": int(num_edges),
        "Number of Components": int(num_components),
        "Largest Component Size": int(largest_component_size),
    }


#download_datasets()

edges = load_edge_list("data/protein.edgelist.txt")
A = create_simple_undirected_adjacency(edges)
metrics = compute_graph_metrics(A)

print(metrics)