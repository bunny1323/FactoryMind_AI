from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Tuple
from backend.config import settings

logger = logging.getLogger("factorymind")

class GraphDatabaseClient:
    def __init__(self):
        self.driver = None
        self.local_graph = {
            "M101": {
                "components": ["Kawasaki K3V112DT Main Pump", "Cummins 6BTAA5.9 Engine", "Turntable Bearing"],
                "relations": [
                    {"source": "M101", "relationship": "HAS_COMPONENT", "target": "Kawasaki K3V112DT Main Pump"},
                    {"source": "Kawasaki K3V112DT Main Pump", "relationship": "HAS_FAULT", "target": "Rotational Shaft Vibration"},
                    {"source": "Rotational Shaft Vibration", "relationship": "RESOLVED_BY", "target": "SOP-MNT-R215-087 Shaft Alignment"},
                    {"source": "SOP-MNT-R215-087 Shaft Alignment", "relationship": "REQUIRES_TOOL", "target": "Laser Shaft Alignment Tool"},
                    {"source": "SOP-MNT-R215-087 Shaft Alignment", "relationship": "REQUIRES_SPARE_PART", "target": "Slewing & Main Pump Bearing Kit (SP-BRG-215)"},
                    {"source": "SOP-MNT-R215-087 Shaft Alignment", "relationship": "HAS_WARNING", "target": "Ensure engine is shut off and pressure relieved"},
                    {"source": "SOP-MNT-R215-087 Shaft Alignment", "relationship": "HAS_MANUAL_SECTION", "target": "Section 4: Pump Component Mounting Torque"},
                    
                    {"source": "M101", "relationship": "HAS_COMPONENT", "target": "Turntable Bearing"},
                    {"source": "Turntable Bearing", "relationship": "HAS_FAULT", "target": "Radial Bearing Clearance Wear"},
                    {"source": "Radial Bearing Clearance Wear", "relationship": "RESOLVED_BY", "target": "Clearance Check & Grease Analysis"},
                    {"source": "Clearance Check & Grease Analysis", "relationship": "REQUIRES_TOOL", "target": "Dial Indicator Gauges"},
                    {"source": "Clearance Check & Grease Analysis", "relationship": "REQUIRES_SPARE_PART", "target": "Turntable Bearing Seal Replacement"},
                    {"source": "Clearance Check & Grease Analysis", "relationship": "HAS_WARNING", "target": "Do not rotate upper structure with gauges attached"},
                    {"source": "Clearance Check & Grease Analysis", "relationship": "HAS_MANUAL_SECTION", "target": "Section 12: Upper Structure Lubrication"},
                ]
            },
            "M102": {
                "components": ["Boom Hydraulic Cylinder", "Oil Return Filters"],
                "relations": [
                    {"source": "M102", "relationship": "HAS_COMPONENT", "target": "Boom Hydraulic Cylinder"},
                    {"source": "Boom Hydraulic Cylinder", "relationship": "HAS_FAULT", "target": "Pressure Seal Leakage"},
                    {"source": "Pressure Seal Leakage", "relationship": "RESOLVED_BY", "target": "Boom Cylinder Seal Replacement SOP"},
                    {"source": "Boom Cylinder Seal Replacement SOP", "relationship": "REQUIRES_TOOL", "target": "Seal Puller Wrench"},
                    {"source": "Boom Cylinder Seal Replacement SOP", "relationship": "REQUIRES_SPARE_PART", "target": "Boom Cylinder Seal Kit (SP-SEL-401)"},
                    {"source": "Boom Cylinder Seal Replacement SOP", "relationship": "HAS_WARNING", "target": "Support boom structure using support stands"},
                    {"source": "Boom Cylinder Seal Replacement SOP", "relationship": "HAS_MANUAL_SECTION", "target": "Section 8: Cylinder Disassembly & Reassembly"}
                ]
            },
            "M103": {
                "components": ["Slew Motor Casing", "Main Control Valve"],
                "relations": [
                    {"source": "M103", "relationship": "HAS_COMPONENT", "target": "Slew Motor Casing"},
                    {"source": "Slew Motor Casing", "relationship": "HAS_FAULT", "target": "Slew Motor Seal Leakage"},
                    {"source": "Slew Motor Seal Leakage", "relationship": "RESOLVED_BY", "target": "Slew Motor Reseal SOP"},
                    {"source": "Slew Motor Reseal SOP", "relationship": "REQUIRES_TOOL", "target": "Standard Wrench Set"},
                    {"source": "Slew Motor Reseal SOP", "relationship": "REQUIRES_SPARE_PART", "target": "Slew Motor O-ring kit"},
                    {"source": "Slew Motor Reseal SOP", "relationship": "HAS_WARNING", "target": "Ensure hydraulic casing is depressurized"},
                    {"source": "Slew Motor Reseal SOP", "relationship": "HAS_MANUAL_SECTION", "target": "Section 5: Slew Drive System"}
                ]
            }
        }
        self.connect()

    def connect(self):
        if settings.NEO4J_URI:
            try:
                from neo4j import GraphDatabase
                logger.info(f"Connecting to Neo4j database at {settings.NEO4J_URI}...")
                self.driver = GraphDatabase.driver(
                    settings.NEO4J_URI, 
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                )
                self.driver.verify_connectivity()
                logger.info("Successfully verified connection to Neo4j Database.")
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}. Falling back to in-memory graph dictionary.")
                self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def ingest_triples(self, triples: list[tuple[str, str, str, str, str]]) -> int:
        """Ingests (src_label, src_name, rel, dst_label, dst_name) triples."""
        if not self.driver:
            logger.info("Local In-Memory Mode: Simulating graph triples ingestion.")
            return len(triples)
            
        cypher = """
        MERGE (a:Label {name: $src_name})
        SET a.type = $src_label
        MERGE (b:Label {name: $dst_name})
        SET b.type = $dst_label
        WITH a, b
        CALL apoc.create.relationship(a, $rel_type, {}, b) YIELD rel
        RETURN count(rel)
        """
        
        count = 0
        with self.driver.session() as session:
            for src_label, src_name, rel, dst_label, dst_name in triples:
                try:
                    # Clean label query (dynamic labels require parameterization precautions or APOC helper)
                    # We will use basic cypher with node type properties
                    session.run(
                        "MERGE (aNode {name: $src_name, label: $src_label}) "
                        "MERGE (bNode {name: $dst_name, label: $dst_label}) "
                        "MERGE (aNode)-[r:RELATED {type: $rel}]->(bNode)",
                        src_name=src_name, src_label=src_label,
                        dst_name=dst_name, dst_label=dst_label,
                        rel=rel
                    )
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to ingest graph edge: {e}")
        return count

    def get_path_for_query(self, query: str, machine_id: str) -> list[dict[str, Any]]:
        """Queries database for path context, defaulting to local dictionary if offline."""
        query_lower = query.lower()
        
        # 1. Query Neo4j if driver is active
        if self.driver:
            try:
                with self.driver.session() as session:
                    # Traverses up to 3 relationship levels: Machine -> Component -> Failure -> Repair
                    result = session.run(
                        "MATCH (m {name: $machine_id})-[r:RELATED]->(c) "
                        "RETURN m.name AS source, r.type AS relationship, c.name AS target "
                        "UNION "
                        "MATCH (m {name: $machine_id})-[:RELATED]->(c)-[r:RELATED]->(f) "
                        "RETURN c.name AS source, r.type AS relationship, f.name AS target "
                        "UNION "
                        "MATCH (m {name: $machine_id})-[:RELATED]->(c)-[:RELATED]->(f)-[r:RELATED]->(rep) "
                        "RETURN f.name AS source, r.type AS relationship, rep.name AS target",
                        machine_id=machine_id
                    )
                    records = []
                    for record in result:
                        records.append({
                            "source": record["source"],
                            "relationship": record["relationship"],
                            "target": record["target"]
                        })
                    if records:
                        logger.info(f"Retrieved {len(records)} relationships from Neo4j DB.")
                        return records
            except Exception as e:
                logger.error(f"Failed to query Neo4j for path: {e}. Falling back to local graph dictionary.")
        
        # 2. Local lookup fallback
        relations = self.local_graph.get(machine_id, self.local_graph["M101"])["relations"]
        
        # Filter relations based on query keywords
        if "vibration" in query_lower or "pump" in query_lower or "coupling" in query_lower:
            return [r for r in relations if "Pump" in r["target"] or "Vibration" in r["target"] or "Alignment" in r["target"] or r["source"] == machine_id]
        elif "bearing" in query_lower or "wear" in query_lower:
            return [r for r in relations if "Bearing" in r["target"] or "Clearance" in r["target"] or "Check" in r["target"] or r["source"] == machine_id]
        elif "leak" in query_lower or "seal" in query_lower:
            return [r for r in relations if "Cylinder" in r["target"] or "Leakage" in r["target"] or "Replacement" in r["target"] or r["source"] == machine_id]
            
        return relations[:4]

graph_client = GraphDatabaseClient()
