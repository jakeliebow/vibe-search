#!/usr/bin/env python3
"""
Louvain Clustering Module
Performs Louvain community detection on graph data from the database.
Also organizes media files into ./communities/<community_idx>/ folders,
renaming them to "{type}_{id}{ext}".
"""

import sys
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import shutil
import pathlib

import networkx as nx
from networkx.algorithms import community

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.psql import PostgresStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_nodes(db) -> List[Dict[str, Any]]:
    """Fetch all nodes from the database"""
    if db.connection is None:
        raise RuntimeError("Database connection not established")

    query = "SELECT id, type, media_path FROM node"
    with db.connection.cursor() as cursor:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    return [dict(zip(columns, row)) for row in rows]

def fetch_edges(db) -> List[Dict[str, Any]]:
    """Fetch all edges from the database"""
    if db.connection is None:
        raise RuntimeError("Database connection not established")

    query = "SELECT id, v1, v2, weight FROM edge"
    with db.connection.cursor() as cursor:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    return [dict(zip(columns, row)) for row in rows]

def build_graph(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
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

def safe_copy(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    """
    Copy src -> dst. If dst exists, append _1, _2, ... before the suffix.
    Returns the final dst path used.
    """
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    stem, suffix = dst.stem, dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1

def organize_media_by_communities(
    nodes: List[Dict[str, Any]],
    communities: List[set],
    base_dir: Optional[str] = None,
) -> None:
    """
    Create ./communities/<idx>/ and copy each node's media_path into the
    corresponding folder, renamed to "{type}_{id}{ext}".
    """
    base = pathlib.Path(base_dir or os.getcwd()) / "communities"
    base.mkdir(parents=True, exist_ok=True)

    # Map node_id (as string) -> node dict for quick lookup
    node_index: Dict[str, Dict[str, Any]] = {str(n["id"]): n for n in nodes}

    for idx, comm in enumerate(communities):
        out_dir = base / str(idx)
        out_dir.mkdir(parents=True, exist_ok=True)

        for node_id in comm:
            n = node_index.get(str(node_id))
            if not n:
                logger.warning(f"Node {node_id} not found in node_index; skipping")
                continue

            media_path = n.get("media_path")
            if not media_path:
                logger.debug(f"Node {node_id} has no media_path; skipping")
                continue

            src = pathlib.Path(media_path)
            if not src.exists() or not src.is_file():
                logger.warning(f"Media file missing for node {node_id}: {src}")
                continue

            node_type = str(n.get("type", "unknown"))
            # Keep original extension (if any)
            ext = src.suffix
            dst_name = f"{node_type}_{n['id']}{ext}"
            dst = out_dir / dst_name

            final_dst = safe_copy(src, dst)
            logger.info(f"[comm {idx}] {src} -> {final_dst.name}")

if __name__ == "__main__":
    with PostgresStorage() as db:
        nodes = fetch_nodes(db)
        edges = fetch_edges(db)
        graph = build_graph(nodes, edges)

        # Louvain community detection
        communities = community.louvain_communities(graph, resolution=0.5, weight="weight")
        logger.info(f"Detected {len(communities)} communities")

        # Organize media into ./communities/<idx>/
        organize_media_by_communities(nodes, communities)
        logger.info("Done.")
