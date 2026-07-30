"""
FactoryMind AI — Lightweight In-Memory Knowledge Graph
Replaces Neo4j with zero external database dependencies.
Fast, reliable, and completely held in RAM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("factorymind")

class InMemoryKnowledgeGraph:
    """Lightweight in-memory entity & relationship store."""

    def __init__(self):
        self.graph: Dict[str, Dict[str, Any]] = {
            "Hyundai R215L Smart Plus": {
                "type": "Machine",
                "components": [
                    "Kawasaki K3V112DT Main Pump", "Cummins 6BTAA5.9 Engine",
                    "Turntable Bearing", "Main Control Valve", "Engine Oil Filter"
                ],
                "failure_modes": ["Excessive Structural Vibration", "Boom Drift", "Overheating"],
                "maintenance_steps": ["500-hour Oil Change", "Slewing Ring Lubrication", "Coupling Alignment"],
                "referenced_pages": [1, 2, 4, 12]
            },
            "Kawasaki K3V112DT Main Pump": {
                "type": "Pump",
                "related_components": ["Main Control Valve", "Engine Flywheel", "Coupling Insert"],
                "failure_modes": ["Bearing degradation", "Coupling insert wear", "Cavitation"],
                "maintenance_steps": ["Measure vibration amplitude", "Inspect axial play", "Replace coupling"],
                "referenced_pages": [7, 8, 24]
            },
            "Cummins 6BTAA5.9 Engine": {
                "type": "Engine",
                "related_components": ["Engine Oil Filter", "Fuel Injectors", "Flywheel"],
                "failure_modes": ["Low oil pressure", "High coolant temperature"],
                "maintenance_steps": ["SAE 15W-40 Oil replacement", "Filter swap"],
                "referenced_pages": [4, 5, 18]
            }
        }
        self.triples: List[Dict[str, str]] = [
            {"source": "Hyundai R215L Smart Plus", "relationship": "connected_to", "target": "Cummins 6BTAA5.9 Engine"},
            {"source": "Cummins 6BTAA5.9 Engine", "relationship": "requires", "target": "SAE 15W-40 Engine Oil"},
            {"source": "Cummins 6BTAA5.9 Engine", "relationship": "maintained_by", "target": "Engine Oil Filter (SP-FLT-101)"},
            {"source": "Hyundai R215L Smart Plus", "relationship": "connected_to", "target": "Kawasaki K3V112DT Main Pump"},
            {"source": "Kawasaki K3V112DT Main Pump", "relationship": "located_in", "target": "Hydraulic Circuit"},
            {"source": "Rotational Shaft Vibration", "relationship": "causes", "target": "Main Pump Coupling Damage"}
        ]
        logger.info("✅ InMemoryKnowledgeGraph initialized successfully (No Neo4j required)")

    def add_entity(self, entity_name: str, entity_type: str, details: Dict[str, Any] = None):
        """Add or update entity node in memory."""
        node = self.graph.setdefault(entity_name, {
            "type": entity_type,
            "related_components": [],
            "failure_modes": [],
            "maintenance_steps": [],
            "referenced_pages": []
        })
        if details:
            for k, v in details.items():
                if isinstance(v, list):
                    node.setdefault(k, []).extend(v)
                else:
                    node[k] = v

    def add_relationship(self, source: str, relationship: str, target: str):
        """Add relationship triple."""
        triple = {"source": source, "relationship": relationship, "target": target}
        if triple not in self.triples:
            self.triples.append(triple)

    def query_graph(self, query_text: str) -> Dict[str, Any]:
        """Retrieve relevant graph context for a query."""
        query_lower = query_text.lower()
        matched_nodes = {}
        matched_relations = []

        for entity, data in self.graph.items():
            if entity.lower() in query_lower or any(word in entity.lower() for word in query_lower.split() if len(word) > 3):
                matched_nodes[entity] = data

        for triple in self.triples:
            if triple["source"].lower() in query_lower or triple["target"].lower() in query_lower:
                matched_relations.append(triple)

        return {
            "nodes": matched_nodes,
            "relationships": matched_relations[:10]
        }

    def close(self):
        """No-op for in-memory graph."""
        pass


class GraphDatabaseClient(InMemoryKnowledgeGraph):
    """Backward-compatible alias for existing imports."""

    def is_connected(self) -> bool:
        return True

    def get_machine_subgraph(self, machine_id: str = "M101") -> Tuple[List[str], List[Dict[str, str]]]:
        nodes = list(self.graph.keys())
        return nodes, self.triples

    def query_subgraph_by_keywords(self, keywords: List[str], max_nodes: int = 15) -> Dict[str, Any]:
        return self.query_graph(" ".join(keywords))

    def get_path_for_query(self, query: str, machine_id: str = "M101") -> List[Dict[str, Any]]:
        """Retrieve matching graph relationship triples for a query."""
        res = self.query_graph(query)
        return res.get("relationships", [])


graph_client = GraphDatabaseClient()
