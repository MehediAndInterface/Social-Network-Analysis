# Temporal Community Detection and User Interaction Analysis

# -----------------------------------------------------------------------------
# 0. IMPORTS AND SETUP
# -----------------------------------------------------------------------------
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set interactive backend (change to 'Agg' if you don't want interactive plots)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict, Counter
import community as community_louvain  # python-louvain package
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import os
import time
import traceback

# --- Configuration ---
DATA_FILEPATH = os.path.join(os.path.dirname(__file__), "CollegeMsg.txt")
NUM_SNAPSHOTS = 10
MIN_NODES_FOR_SPECTRAL = 10
N_COMMUNITIES_SPECTRAL = 5
OUTPUT_DIR = "sna_project_output"
JACCARD_THRESHOLD = 0.3
INTERACTIVE_VISUALIZATION = True

# Validate configuration
if NUM_SNAPSHOTS < 1:
    print("ERROR: NUM_SNAPSHOTS must be at least 1")
    exit(1)

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        print(f"ERROR: Could not create output directory: {str(e)}")
        exit(1)


# -----------------------------------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------
def load_and_preprocess_data(filepath, num_snapshots):
    """Loads and divides dataset into temporal snapshots."""
    print(f"\nLoading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=' ', header=None,
                         names=['source', 'target', 'timestamp'])
    except Exception as e:
        print(f"ERROR: Could not load data: {str(e)}")
        return None, None

    print(f"Loaded {len(df)} interactions.")
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    total_time_span = max_time - min_time

    if total_time_span == 0:
        if len(df) > 0:
            print("Warning: All events have same timestamp. Using single snapshot.")
            time_boundaries = [min_time, max_time + 1]
            actual_num_snapshots = 1
        else:
            print("Error: No data to process.")
            return None, None
    else:
        snapshot_duration = total_time_span / num_snapshots
        time_boundaries = [min_time + i * snapshot_duration for i in range(num_snapshots + 1)]
        time_boundaries[-1] = max_time + 1
        actual_num_snapshots = num_snapshots

    print(f"\nCreating {actual_num_snapshots} snapshots...")
    snapshot_graphs = []
    snapshot_data_info = []

    for i in range(actual_num_snapshots):
        try:
            snapshot_df = df[(df['timestamp'] >= time_boundaries[i]) &
                             (df['timestamp'] < time_boundaries[i + 1])]

            G = nx.from_pandas_edgelist(snapshot_df, 'source', 'target',
                                        create_using=nx.Graph()) if not snapshot_df.empty else nx.Graph()

            snapshot_graphs.append(G)
            snapshot_data_info.append({
                'id': i,
                'start_time': time_boundaries[i],
                'end_time': time_boundaries[i + 1],
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges()
            })

            print(f"Snapshot {i + 1}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        except Exception as e:
            print(f"ERROR processing snapshot {i + 1}: {str(e)}")
            snapshot_graphs.append(nx.Graph())
            snapshot_data_info.append({
                'id': i,
                'start_time': time_boundaries[i],
                'end_time': time_boundaries[i + 1],
                'num_nodes': 0,
                'num_edges': 0
            })

    return snapshot_graphs, snapshot_data_info


# -----------------------------------------------------------------------------
# 2. COMMUNITY DETECTION ALGORITHMS
# -----------------------------------------------------------------------------
def detect_communities_louvain(graph):
    """Louvain method for community detection."""
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return community_louvain.best_partition(graph, random_state=42)
    except Exception as e:
        print(f"Louvain algorithm failed: {str(e)}")
        return {}


def detect_communities_spectral(graph, num_communities_louvain):
    """Spectral clustering for community detection."""
    if graph.number_of_nodes() == 0:
        return {}

    try:
        k = num_communities_louvain if N_COMMUNITIES_SPECTRAL == 'auto' else N_COMMUNITIES_SPECTRAL
        k = max(2, min(k, graph.number_of_nodes() - 1))

        if not nx.is_connected(graph):
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc).copy()
            if subgraph.number_of_nodes() < max(k, MIN_NODES_FOR_SPECTRAL):
                return {}
            graph_to_cluster = subgraph
        else:
            if graph.number_of_nodes() < max(k, MIN_NODES_FOR_SPECTRAL):
                return {}
            graph_to_cluster = graph

        adj_matrix = nx.to_numpy_array(graph_to_cluster)
        sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                                assign_labels='kmeans', random_state=42)
        labels = sc.fit_predict(adj_matrix)

        partition = {node: labels[i] for i, node in enumerate(graph_to_cluster.nodes())}

        # Handle disconnected components
        if not nx.is_connected(graph):
            max_label = max(labels) if labels.size > 0 else -1
            current_label = max_label + 1
            for node in graph.nodes():
                if node not in partition:
                    partition[node] = current_label
                    current_label += 1

        return partition

    except Exception as e:
        print(f"Spectral clustering failed: {str(e)}")
        return {}


# -----------------------------------------------------------------------------
# 3. VISUALIZATION FUNCTIONS (UPDATED FOR INTERACTIVE DISPLAY)
# -----------------------------------------------------------------------------
def visualize_snapshot_communities(graph, partition, snapshot_index, algorithm_name="louvain"):
    """Visualizes graph with communities, shows interactively and saves to file."""
    if graph.number_of_nodes() == 0 or not partition:
        print(f"Snapshot {snapshot_index}: No graph/partition to visualize")
        return

    if graph.number_of_nodes() > 1000:
        print(f"Snapshot {snapshot_index}: Too large to visualize ({graph.number_of_nodes()} nodes)")
        return

    plt.figure(figsize=(12, 12))
    plt.rcParams['keymap.quit'].append('q')  # Allow quitting with 'q' key

    try:
        # Choose layout based on graph size
        if graph.number_of_nodes() > 500:
            pos = nx.random_layout(graph, seed=42)
        elif graph.number_of_nodes() > 100:
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42, k=0.15, iterations=20)

        # Assign colors to communities
        community_ids = set(partition.values())
        colors = cm.get_cmap('viridis', len(community_ids)) if community_ids else ['blue']
        node_colors = [colors(i % len(colors)) for i in partition.values()]

        # Draw the graph
        nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5)
        nodes = nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                                       node_size=50, alpha=0.8)

        # Add labels for smaller graphs
        if graph.number_of_nodes() <= 100:
            nx.draw_networkx_labels(graph, pos, font_size=8)

        plt.title(f"Snapshot {snapshot_index} - {algorithm_name.capitalize()}\n"
                  f"{graph.number_of_nodes()} Nodes, {graph.number_of_edges()} Edges, "
                  f"{len(community_ids)} Communities")
        plt.axis('off')

        # Save to file
        filename = os.path.join(OUTPUT_DIR, f"snapshot_{snapshot_index}_{algorithm_name}.png")
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved visualization: {filename}")

        # Show interactively if enabled
        if INTERACTIVE_VISUALIZATION:
            plt.show(block=True)  # Blocking mode for proper display
        else:
            plt.close()

    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        plt.close()


# -----------------------------------------------------------------------------
# 4. COMMUNITY EVOLUTION ANALYSIS
# -----------------------------------------------------------------------------
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0


def get_community_sets(partition):
    comm_map = defaultdict(set)
    for node, comm_id in partition.items():
        comm_map[comm_id].add(node)
    return list(comm_map.values())


def analyze_community_evolution(partitions):
    print("\n=== Community Evolution Analysis ===")
    for i in range(len(partitions) - 1):
        comms_t0 = get_community_sets(partitions[i])
        comms_t1 = get_community_sets(partitions[i + 1])

        print(f"\nSnapshot {i} -> {i + 1}:")
        print(f"  Communities: {len(comms_t0)} -> {len(comms_t1)}")

        # Basic similarity analysis
        if comms_t0 and comms_t1:
            avg_jaccard = np.mean([max(jaccard_similarity(c0, c1)
                                       for c1 in comms_t1) for c0 in comms_t0])
            print(f"  Average community similarity: {avg_jaccard:.3f}")


# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Temporal Community Detection ===")

    try:
        # Load and preprocess data
        graphs, info = load_and_preprocess_data(DATA_FILEPATH, NUM_SNAPSHOTS)
        if not graphs:
            exit(1)

        # Community detection
        louvain_partitions = []
        spectral_partitions = []

        print("\n=== Detecting Communities ===")
        for i, G in enumerate(graphs):
            print(f"\nSnapshot {i + 1}/{len(graphs)}:")

            # Louvain
            louvain_part = detect_communities_louvain(G)
            louvain_partitions.append(louvain_part)
            if louvain_part:
                mod = community_louvain.modularity(louvain_part, G)
                print(f"Louvain: {len(set(louvain_part.values()))} communities, modularity {mod:.3f}")
                visualize_snapshot_communities(G, louvain_part, i, "louvain")

            # Spectral (for comparison)
            spectral_part = detect_communities_spectral(G, len(set(louvain_part.values()))) if louvain_part else {}
            spectral_partitions.append(spectral_part)

            if spectral_part:
                mod = community_louvain.modularity(spectral_part, G)
                print(f"Spectral: {len(set(spectral_part.values()))} communities, modularity {mod:.3f}")
                visualize_snapshot_communities(G, spectral_part, i, algorithm_name="spectral")

            # Analysis
            analyze_community_evolution(louvain_partitions)

            print("\n=== Analysis Complete ===")
            print(f"Visualizations saved to: {os.path.abspath(OUTPUT_DIR)}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        traceback.print_exc()
        exit(1)