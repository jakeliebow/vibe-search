#!/usr/bin/env python3
"""
Louvain Clustering Module
Performs Louvain community detection on graph data from the database.
"""

import sys
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.psql import PostgresStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphClusterer:
    """Handles graph clustering using Louvain algorithm"""

    def __init__(self, db_connection: PostgresStorage):
        self.db = db_connection

    def fetch_nodes(self) -> List[Dict[str, Any]]:
        """Fetch all nodes from the database"""
        if self.db.connection is None:
            raise RuntimeError("Database connection not established")

        query = "SELECT id, type FROM node"

        with self.db.connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def fetch_edges(self) -> List[Dict[str, Any]]:
        """Fetch all edges from the database"""
        if self.db.connection is None:
            raise RuntimeError("Database connection not established")

        query = "SELECT id, v1, v2, weight FROM edge"

        with self.db.connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def build_graph(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> nx.Graph:
        """Build NetworkX graph from nodes and edges data"""
        G = nx.Graph()

        # Add nodes with attributes
        for node in nodes:
            G.add_node(str(node["id"]), node_type=node["type"])

        # Add edges with weights
        for edge in edges:
            G.add_edge(str(edge["v1"]), str(edge["v2"]), weight=edge["weight"])

        logger.info(
            f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    def run_louvain_clustering(
        self, G: nx.Graph, resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Run Louvain clustering on the graph

        Args:
            G: NetworkX graph
            resolution: Resolution parameter for clustering (higher = more communities)

        Returns:
            Dictionary mapping node IDs to community IDs
        """
        logger.info(f"Running Louvain clustering with resolution={resolution}")

        # Run Louvain community detection
        communities = community.louvain_communities(
            G, resolution=resolution, weight="weight"
        )

        # Convert to node -> community mapping
        node_to_community = {}
        for community_id, community_nodes in enumerate(communities):
            for node in community_nodes:
                node_to_community[node] = community_id

        logger.info(f"Found {len(communities)} communities")
        return node_to_community

    def cluster_graph(
        self, resolution: float = 1.0
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Main clustering function that fetches data and runs Louvain clustering

        Args:
            resolution: Resolution parameter for clustering

        Returns:
            Tuple of (node_to_community mapping, clustering statistics)
        """
        logger.info("Starting graph clustering process")

        # Fetch data from database
        nodes = self.fetch_nodes()
        edges = self.fetch_edges()

        logger.info(f"Fetched {len(nodes)} nodes and {len(edges)} edges from database")

        if not nodes:
            logger.warning("No nodes found in database")
            return {}, {"num_nodes": 0, "num_edges": 0, "num_communities": 0}

        # Build graph
        G = self.build_graph(nodes, edges)

        if G.number_of_nodes() == 0:
            logger.warning("Graph has no nodes")
            return {}, {"num_nodes": 0, "num_edges": 0, "num_communities": 0}

        # Run clustering
        node_to_community = self.run_louvain_clustering(G, resolution)

        # Calculate statistics
        stats = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_communities": (
                len(set(node_to_community.values())) if node_to_community else 0
            ),
            "resolution": resolution,
        }

        return node_to_community, stats

    def visualize_network(
        self,
        G: nx.Graph,
        node_to_community: Dict[str, int],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        layout_type: str = "spring",
    ):
        """
        Visualize the network graph with community colors

        Args:
            G: NetworkX graph
            node_to_community: Mapping of nodes to communities
            output_path: Path to save the visualization
            figsize: Figure size (width, height)
            layout_type: Layout algorithm ('spring', 'circular', 'random', 'kamada_kawai')
        """
        plt.figure(figsize=figsize)

        # Choose layout algorithm
        if layout_type == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout_type == "circular":
            pos = nx.circular_layout(G)
        elif layout_type == "random":
            pos = nx.random_layout(G)
        elif layout_type == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Get unique communities and assign colors
        communities = list(set(node_to_community.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        community_colors = dict(zip(communities, colors))

        # Color nodes by community
        node_colors = [
            community_colors.get(node_to_community.get(node, 0), "gray")
            for node in G.nodes()
        ]

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

        # Add labels for smaller graphs
        if G.number_of_nodes() <= 50:
            nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(
            f"Network Clustering Visualization\n{len(communities)} Communities, {G.number_of_nodes()} Nodes"
        )
        plt.axis("off")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Network visualization saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_community_stats(
        self,
        node_to_community: Dict[str, int],
        nodes: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
    ):
        """
        Create comprehensive community statistics visualizations

        Args:
            node_to_community: Mapping of nodes to communities
            nodes: List of node data with types
            output_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Community Analysis Dashboard", fontsize=16, fontweight="bold")

        # Prepare data
        community_sizes = Counter(node_to_community.values())
        node_types = {str(node["id"]): node["type"] for node in nodes}

        # 1. Community size distribution
        axes[0, 0].bar(range(len(community_sizes)), list(community_sizes.values()))
        axes[0, 0].set_title("Community Size Distribution")
        axes[0, 0].set_xlabel("Community ID")
        axes[0, 0].set_ylabel("Number of Nodes")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Community size histogram
        sizes = list(community_sizes.values())
        axes[0, 1].hist(
            sizes, bins=max(1, len(sizes) // 3), alpha=0.7, edgecolor="black"
        )
        axes[0, 1].set_title("Community Size Histogram")
        axes[0, 1].set_xlabel("Community Size")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Node type distribution by community
        community_node_types = {}
        for node_id, community_id in node_to_community.items():
            node_type = node_types.get(node_id, "unknown")
            if community_id not in community_node_types:
                community_node_types[community_id] = Counter()
            community_node_types[community_id][node_type] += 1

        # Create stacked bar chart for node types
        if community_node_types:
            all_types = set()
            for type_counter in community_node_types.values():
                all_types.update(type_counter.keys())
            all_types = sorted(list(all_types))

            bottom = np.zeros(len(community_node_types))
            colors = plt.cm.Set2(np.linspace(0, 1, len(all_types)))

            for i, node_type in enumerate(all_types):
                values = [
                    community_node_types[comm_id].get(node_type, 0)
                    for comm_id in sorted(community_node_types.keys())
                ]
                axes[1, 0].bar(
                    range(len(community_node_types)),
                    values,
                    bottom=bottom,
                    label=node_type,
                    color=colors[i],
                )
                bottom += values

            axes[1, 0].set_title("Node Types by Community")
            axes[1, 0].set_xlabel("Community ID")
            axes[1, 0].set_ylabel("Number of Nodes")
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Summary statistics
        stats_text = f"""
        Total Communities: {len(community_sizes)}
        Total Nodes: {len(node_to_community)}
        Largest Community: {max(community_sizes.values()) if community_sizes else 0}
        Smallest Community: {min(community_sizes.values()) if community_sizes else 0}
        Average Community Size: {np.mean(list(community_sizes.values())):.2f}
        Modularity: Computing...
        """

        axes[1, 1].text(
            0.1,
            0.5,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        axes[1, 1].set_title("Summary Statistics")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Community statistics saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def create_community_comparison(
        self,
        resolution_results: Dict[float, Tuple[Dict[str, int], Dict[str, Any]]],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Compare clustering results across different resolution values

        Args:
            resolution_results: Dict mapping resolution values to (communities, stats)
            output_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("Resolution Comparison Analysis", fontsize=16, fontweight="bold")

        resolutions = sorted(resolution_results.keys())
        num_communities = [
            resolution_results[r][1]["num_communities"] for r in resolutions
        ]

        # 1. Number of communities vs resolution
        axes[0].plot(resolutions, num_communities, "bo-", linewidth=2, markersize=8)
        axes[0].set_title("Communities vs Resolution")
        axes[0].set_xlabel("Resolution Parameter")
        axes[0].set_ylabel("Number of Communities")
        axes[0].grid(True, alpha=0.3)

        # 2. Community size distribution comparison
        for i, resolution in enumerate(resolutions[:5]):  # Show up to 5 resolutions
            communities, _ = resolution_results[resolution]
            community_sizes = list(Counter(communities.values()).values())

            # Create histogram data
            if community_sizes:
                axes[1].hist(
                    community_sizes,
                    bins=max(1, max(community_sizes) // 2),
                    alpha=0.6,
                    label=f"Resolution {resolution}",
                    density=True,
                )

        axes[1].set_title("Community Size Distributions")
        axes[1].set_xlabel("Community Size")
        axes[1].set_ylabel("Density")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Resolution comparison saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def export_communities_for_analysis(
        self,
        node_to_community: Dict[str, int],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        output_path: str,
    ):
        """
        Export detailed community analysis to CSV files

        Args:
            node_to_community: Mapping of nodes to communities
            nodes: List of node data
            edges: List of edge data
            output_path: Base path for output files (without extension)
        """
        import csv

        # Export node-community mapping
        with open(f"{output_path}_nodes.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "node_type", "community_id"])

            node_types = {str(node["id"]): node["type"] for node in nodes}
            for node_id, community_id in node_to_community.items():
                node_type = node_types.get(node_id, "unknown")
                writer.writerow([node_id, node_type, community_id])

        # Export community statistics
        community_stats = Counter(node_to_community.values())
        with open(f"{output_path}_communities.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["community_id", "size", "percentage"])

            total_nodes = len(node_to_community)
            for community_id, size in community_stats.items():
                percentage = (size / total_nodes) * 100
                writer.writerow([community_id, size, f"{percentage:.2f}%"])

        # Export edges with community information
        with open(f"{output_path}_edges.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "edge_id",
                    "node1",
                    "node2",
                    "weight",
                    "community1",
                    "community2",
                    "inter_community",
                ]
            )

            for edge in edges:
                node1, node2 = str(edge["v1"]), str(edge["v2"])
                comm1 = node_to_community.get(node1, -1)
                comm2 = node_to_community.get(node2, -1)
                inter_community = comm1 != comm2

                writer.writerow(
                    [
                        edge["id"],
                        node1,
                        node2,
                        edge["weight"],
                        comm1,
                        comm2,
                        inter_community,
                    ]
                )

        logger.info(f"Community analysis exported to {output_path}_*.csv")


def main():
    """Main function to run clustering with configurable resolution and visualization"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Louvain clustering on graph data with visualization"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for clustering (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save clustering results (optional)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )
    parser.add_argument(
        "--viz-network",
        type=str,
        default=None,
        help="Save network visualization to file",
    )
    parser.add_argument(
        "--viz-stats",
        type=str,
        default=None,
        help="Save community statistics dashboard to file",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "circular", "random", "kamada_kawai"],
        help="Network layout algorithm (default: spring)",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export detailed analysis to CSV files (base filename)",
    )
    parser.add_argument(
        "--compare-resolutions",
        nargs="+",
        type=float,
        default=None,
        help="Compare multiple resolution values (e.g., --compare-resolutions 0.5 1.0 1.5 2.0)",
    )
    parser.add_argument(
        "--viz-comparison",
        type=str,
        default=None,
        help="Save resolution comparison visualization to file",
    )

    args = parser.parse_args()

    try:
        # Initialize database connection
        logger.info("Connecting to database...")
        with PostgresStorage() as db:
            clusterer = GraphClusterer(db)

            # Handle resolution comparison mode
            if args.compare_resolutions:
                logger.info(f"Comparing resolutions: {args.compare_resolutions}")
                resolution_results = {}

                for resolution in args.compare_resolutions:
                    logger.info(f"Running clustering with resolution={resolution}")
                    node_to_community, stats = clusterer.cluster_graph(
                        resolution=resolution
                    )
                    resolution_results[resolution] = (node_to_community, stats)

                # Print comparison results
                print(f"\nResolution Comparison Results:")
                print("-" * 50)
                for resolution in sorted(resolution_results.keys()):
                    _, stats = resolution_results[resolution]
                    print(
                        f"Resolution {resolution:4.1f}: {stats['num_communities']:3d} communities, "
                        f"{stats['num_nodes']:3d} nodes, {stats['num_edges']:3d} edges"
                    )

                # Create comparison visualization
                if args.visualize or args.viz_comparison:
                    clusterer.create_community_comparison(
                        resolution_results, output_path=args.viz_comparison
                    )

                return

            # Single resolution mode
            logger.info(f"Running clustering with resolution={args.resolution}")

            # Fetch data for visualization
            nodes = clusterer.fetch_nodes()
            edges = clusterer.fetch_edges()

            if not nodes:
                logger.warning("No data available for clustering")
                return

            # Build graph for visualization
            G = clusterer.build_graph(nodes, edges)

            # Run clustering
            node_to_community, stats = clusterer.cluster_graph(
                resolution=args.resolution
            )

            # Print results
            print(f"\nClustering Results:")
            print(f"Resolution: {stats['resolution']}")
            print(f"Nodes: {stats['num_nodes']}")
            print(f"Edges: {stats['num_edges']}")
            print(f"Communities found: {stats['num_communities']}")

            if node_to_community:
                print(f"\nSample community assignments:")
                for i, (node_id, community_id) in enumerate(
                    list(node_to_community.items())[:10]
                ):
                    print(f"  Node {node_id}: Community {community_id}")
                if len(node_to_community) > 10:
                    print(f"  ... and {len(node_to_community) - 10} more")

            # Generate visualizations
            if args.visualize or args.viz_network or args.viz_stats:
                logger.info("Generating visualizations...")

                # Network visualization
                if args.visualize or args.viz_network:
                    clusterer.visualize_network(
                        G,
                        node_to_community,
                        output_path=args.viz_network,
                        layout_type=args.layout,
                    )

                # Community statistics dashboard
                if args.visualize or args.viz_stats:
                    clusterer.visualize_community_stats(
                        node_to_community, nodes, output_path=args.viz_stats
                    )

            # Export CSV analysis
            if args.export_csv:
                logger.info("Exporting detailed analysis to CSV...")
                clusterer.export_communities_for_analysis(
                    node_to_community, nodes, edges, args.export_csv
                )

            # Save JSON results
            if args.output:
                import json

                output_data = {
                    "statistics": stats,
                    "node_to_community": node_to_community,
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2, default=str)
                logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error during clustering: {e}")
        raise


if __name__ == "__main__":
    main()
