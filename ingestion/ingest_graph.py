from __future__ import annotations

import os
import logging
from graph.neo4j_client import graph_client
from backend.config import settings

logger = logging.getLogger("factorymind")

def run_graph_ingestion() -> int:
    """Extracts component-failure-repair associations from R215L docs and ingests them.
    Note: This is a heuristic/rule-based parser based on keyword matching, not a full Named Entity Recognition (NER) model.
    """
    logger.info("Extracting machinery semantic associations for Neo4j...")
    
    manuals_dir = os.path.join(settings.DATA_DIR, "manuals")
    sop_dir = os.path.join(settings.DATA_DIR, "sop")
    
    comp_keywords = ["pump", "bearing", "cylinder", "motor", "flywheel", "coupling", "valve", "filter"]
    fail_keywords = ["vibration", "leak", "wear", "noise", "high temperature", "overstrain", "pressure drop", "play"]
    repair_keywords = ["alignment", "replacement", "reseal", "lubrication", "torque check", "inspection"]
    
    extracted_triples = []
    
    # Heuristic processing function
    def extract_from_text(text: str, filename: str):
        # We split text into sentences/lines
        sentences = text.split(".")
        for s in sentences:
            s_lower = s.lower()
            # Look for machine name if any, e.g. M101, M102, M103
            machine = "M101"
            if "m102" in s_lower:
                machine = "M102"
            elif "m103" in s_lower:
                machine = "M103"
                
            # Find matching component
            found_comp = None
            for ck in comp_keywords:
                if ck in s_lower:
                    # Map to canonical name
                    if ck == "pump":
                        found_comp = "Kawasaki K3V112DT Main Pump"
                    elif ck == "bearing":
                        found_comp = "Turntable Bearing"
                    elif ck == "cylinder":
                        found_comp = "Boom Hydraulic Cylinder"
                    elif ck == "motor":
                        found_comp = "Slew Motor Casing"
                    elif ck == "coupling":
                        found_comp = "Engine-Pump Coupling Insert"
                    else:
                        found_comp = ck.capitalize()
                    break
            
            # Find matching failure
            found_fail = None
            for fk in fail_keywords:
                if fk in s_lower:
                    if fk == "vibration":
                        found_fail = "Rotational Shaft Vibration"
                    elif fk == "leak":
                        found_fail = "Pressure Seal Leakage"
                    elif fk == "wear":
                        found_fail = "Radial Bearing Clearance Wear"
                    else:
                        found_fail = fk.capitalize()
                    break
                    
            # Find matching repair
            found_repair = None
            for rk in repair_keywords:
                if rk in s_lower:
                    if rk == "alignment":
                        found_repair = "SOP-MNT-R215-087 Shaft Alignment"
                    elif rk == "replacement":
                        found_repair = "Component Replacement"
                    elif rk == "reseal":
                        found_repair = "Cylinder Reseal SOP"
                    else:
                        found_repair = rk.capitalize()
                    break
            
            # If we found matches, build triples
            if found_comp:
                extracted_triples.append(("Machine", machine, "HAS_COMPONENT", "Component", found_comp))
                if found_fail:
                    extracted_triples.append(("Component", found_comp, "CAN_FAIL_AS", "Failure", found_fail))
                    if found_repair:
                        extracted_triples.append(("Failure", found_fail, "RESOLVED_BY", "Repair", found_repair))

    # Load and process manuals
    if os.path.exists(manuals_dir):
        for f in os.listdir(manuals_dir):
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(manuals_dir, f), "r", encoding="utf-8") as file:
                        extract_from_text(file.read(), f)
                except Exception as e:
                    logger.error(f"Error reading {f} for graph extraction: {e}")

    # Load and process SOPs
    if os.path.exists(sop_dir):
        for f in os.listdir(sop_dir):
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(sop_dir, f), "r", encoding="utf-8") as file:
                        extract_from_text(file.read(), f)
                except Exception as e:
                    logger.error(f"Error reading {f} for graph extraction: {e}")

    # Remove duplicates while preserving order
    seen = set()
    unique_triples = []
    for t in extracted_triples:
        if t not in seen:
            seen.add(t)
            unique_triples.append(t)
            
    # Fallback to default static triples if heuristic extracted nothing or files were missing
    if not unique_triples:
        logger.info("Heuristic extraction found no matches or files are missing. Falling back to default static triples.")
        unique_triples = [
            ("Machine", "M101", "HAS_COMPONENT", "Component", "Kawasaki K3V112DT Main Pump"),
            ("Component", "Kawasaki K3V112DT Main Pump", "CAN_FAIL_AS", "Failure", "Rotational Shaft Vibration"),
            ("Failure", "Rotational Shaft Vibration", "RESOLVED_BY", "Repair", "SOP-MNT-R215-087 Shaft Alignment"),
            ("Failure", "Rotational Shaft Vibration", "REQUIRES_PART", "SparePart", "SP-CPL-332 Coupling Insert"),
            ("Machine", "M101", "HAS_COMPONENT", "Component", "Turntable Bearing"),
            ("Component", "Turntable Bearing", "CAN_FAIL_AS", "Failure", "Radial Bearing Clearance Wear"),
            ("Failure", "Radial Bearing Clearance Wear", "RESOLVED_BY", "Repair", "Grease Analysis & Clearance Check"),
            ("Failure", "Radial Bearing Clearance Wear", "REQUIRES_PART", "SparePart", "SP-BRG-215 Bearing Kit"),
            ("Machine", "M102", "HAS_COMPONENT", "Component", "Boom Hydraulic Cylinder"),
            ("Component", "Boom Hydraulic Cylinder", "CAN_FAIL_AS", "Failure", "Pressure Seal Leakage"),
            ("Failure", "Pressure Seal Leakage", "RESOLVED_BY", "Repair", "Cylinder Reseal SOP"),
            ("Failure", "Pressure Seal Leakage", "REQUIRES_PART", "SparePart", "SP-SEL-401 Seal Kit"),
            ("Machine", "M103", "HAS_COMPONENT", "Component", "Slew Motor Casing"),
            ("Component", "Slew Motor Casing", "CAN_FAIL_AS", "Failure", "Slew Motor Seal Leakage"),
            ("Failure", "Slew Motor Seal Leakage", "RESOLVED_BY", "Repair", "Slew Motor Reseal SOP"),
            ("Failure", "Slew Motor Seal Leakage", "REQUIRES_PART", "SparePart", "SP-SEL-401 Seal Kit")
        ]
        
    count = graph_client.ingest_triples(unique_triples)
    logger.info(f"Successfully indexed {count} triples in Graph database.")
    return count
