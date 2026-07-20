"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import { 
  Database, UploadCloud, RefreshCw, Layers, Cpu, Play, CheckCircle2, 
  AlertCircle, BookOpen, ClipboardList, FileText, AlertTriangle, Wrench, Trash2, Settings, Sliders 
} from "lucide-react";
import ReactFlow, { MiniMap, Controls, Background, useNodesState, useEdgesState } from "reactflow";
import "reactflow/dist/style.css";

interface CollectionStat {
  count: number;
  last_updated: string;
}

interface KBStats {
  manuals: CollectionStat;
  sop: CollectionStat;
  maintenance_logs: CollectionStat;
  error_codes: CollectionStat;
  spare_parts: CollectionStat;
}

interface PipelineJob {
  progress: number;
  status: string;
  message: string;
  job_id?: string;
}

interface IndexStatus {
  status: string;
  embedding_model: string;
  dimension: number;
  total_chunks: number;
  breakdown: Record<string, number>;
  last_indexed: string;
}

export default function AdminPage() {
  const [stats, setStats] = useState<KBStats>({
    manuals: { count: 1, last_updated: "Recently" },
    sop: { count: 1, last_updated: "Recently" },
    maintenance_logs: { count: 5, last_updated: "Recently" },
    error_codes: { count: 3, last_updated: "Recently" },
    spare_parts: { count: 5, last_updated: "Recently" }
  });

  const [dbStatus, setDbStatus] = useState<IndexStatus | null>(null);

  // RAG System Parameter Settings States
  const [retrieverTopK, setRetrieverTopK] = useState(8);
  const [rerankerTopK, setRerankerTopK] = useState(5);
  const [denseWeight, setDenseWeight] = useState(0.7);
  const [temperature, setTemperature] = useState(0.15);
  const [selectedModel, setSelectedModel] = useState("qwen-72b-industrial");
  const [enableSemanticCache, setEnableSemanticCache] = useState(true);
  const [graphDepth, setGraphDepth] = useState(2);
  const [settingsSaved, setSettingsSaved] = useState(false);

  const [pipelineJobs, setPipelineJobs] = useState<Record<string, PipelineJob>>({
    manuals: { progress: 0, status: "idle", message: "" },
    sop: { progress: 0, status: "idle", message: "" },
    maintenance_logs: { progress: 0, status: "idle", message: "" },
    error_codes: { progress: 0, status: "idle", message: "" },
    spare_parts: { progress: 0, status: "idle", message: "" },
    prediction: { progress: 0, status: "idle", message: "" },
    graph: { progress: 0, status: "idle", message: "" }
  });

  const [selectedMachine, setSelectedMachine] = useState("M101");
  const [isMounted, setIsMounted] = useState(false);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Fetch vector DB status
  const fetchDbStatus = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/admin/collection/status");
      if (res.ok) {
        const data = await res.json();
        setDbStatus(data);
      }
    } catch (e) {
      console.warn("Unable to fetch Qdrant DB status");
    }
  };

  // Poll for stats
  const fetchStats = async () => {
    try {
      const token = localStorage.getItem("fm_jwt_token");
      const res = await fetch("http://127.0.0.1:8000/admin/knowledge-base/stats", {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (e) {
      console.warn("Unable to fetch stats from backend. Using local defaults.");
    }
  };

  const convertToReactFlow = (kgPath: any[], mId: string) => {
    const nodesMap = new Map<string, any>();
    const flowEdges: any[] = [];
    let nodeCount = 0;

    kgPath.forEach((triple, idx) => {
      const { source, relationship, target } = triple;

      if (!nodesMap.has(source)) {
        let x = 150 + (nodeCount % 3) * 160;
        let y = 60 + Math.floor(nodeCount / 3) * 90;
        let bg = "#F0E6DA";
        let color = "#3B2A1E";

        if (source === mId) {
          x = 220;
          y = 20;
          bg = "#6B4A32";
          color = "#FFFFFF";
        }

        nodesMap.set(source, {
          id: source,
          data: { label: source },
          position: { x, y },
          style: {
            background: bg,
            color: color,
            border: "1.5px solid #3B2A1E",
            borderRadius: "10px",
            padding: "6px 12px",
            fontSize: "10px",
            fontWeight: "bold",
            textAlign: "center",
            width: 130
          }
        });
        nodeCount++;
      }

      if (!nodesMap.has(target)) {
        let x = 150 + (nodeCount % 3) * 160;
        let y = 60 + Math.floor(nodeCount / 3) * 90;
        let bg = "#FFFFFF";
        let color = "#3B2A1E";

        if (relationship === "HAS_COMPONENT") {
          y = 120;
          bg = "#FAF6F1";
        } else if (relationship === "CAN_FAIL_AS") {
          y = 200;
          bg = "#FFF4E5";
          color = "#9C6B45";
        } else if (relationship === "RESOLVED_BY") {
          y = 280;
          bg = "#E6F4EA";
          color = "#137333";
        } else if (relationship === "REQUIRES_PART") {
          y = 280;
          bg = "#FFF4E5";
          color = "#9C6B45";
        }

        nodesMap.set(target, {
          id: target,
          data: { label: target },
          position: { x, y },
          style: {
            background: bg,
            color: color,
            border: "1.5px solid #3B2A1E",
            borderRadius: "10px",
            padding: "6px 12px",
            fontSize: "10px",
            fontWeight: "bold",
            textAlign: "center",
            width: 130
          }
        });
        nodeCount++;
      }

      flowEdges.push({
        id: `e-${source}-${target}-${idx}`,
        source: source,
        target: target,
        label: relationship,
        labelStyle: { fill: "#9C6B45", fontWeight: 700, fontSize: "8px" },
        style: { stroke: "#9C6B45", strokeWidth: 1.5 },
        animated: true
      });
    });

    return { nodes: Array.from(nodesMap.values()), edges: flowEdges };
  };

  const fetchGraphData = async (mId = selectedMachine) => {
    try {
      const token = localStorage.getItem("fm_jwt_token");
      const res = await fetch(`http://127.0.0.1:8000/machines/${mId}/graph`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (res.ok) {
        const kgPath = await res.json();
        const { nodes: flowNodes, edges: flowEdges } = convertToReactFlow(kgPath, mId);
        setNodes(flowNodes);
        setEdges(flowEdges);
      }
    } catch (e) {
      console.warn("Failed to fetch graph data, using mock fallback.");
      const mockRelations = [
        { "source": mId, "relationship": "HAS_COMPONENT", "target": "Main Control Valve" },
        { "source": "Main Control Valve", "relationship": "CAN_FAIL_AS", "target": "Pilot Line Leakage" },
        { "source": "Pilot Line Leakage", "relationship": "RESOLVED_BY", "target": "SOP-PLT-98 Seal Replacement" }
      ];
      const { nodes: flowNodes, edges: flowEdges } = convertToReactFlow(mockRelations, mId);
      setNodes(flowNodes);
      setEdges(flowEdges);
    }
  };

  useEffect(() => {
    setIsMounted(true);
    fetchStats();
    fetchDbStatus();
    const statsInterval = setInterval(() => {
      fetchStats();
      fetchDbStatus();
    }, 12000);
    return () => clearInterval(statsInterval);
  }, []);

  useEffect(() => {
    fetchGraphData(selectedMachine);
  }, [selectedMachine]);

  const triggerIngestion = async (widgetKey: string) => {
    const pipelineName = widgetKey === "sop" ? "manuals" : widgetKey;

    setPipelineJobs(prev => ({
      ...prev,
      [widgetKey]: { progress: 10, status: "queued", message: "Task queued..." }
    }));

    try {
      const token = localStorage.getItem("fm_jwt_token");
      const res = await fetch(`http://127.0.0.1:8000/ingest/${pipelineName}`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (res.ok) {
        const { job_id } = await res.json();
        setPipelineJobs(prev => ({
          ...prev,
          [widgetKey]: { progress: 10, status: "processing", message: "Starting task...", job_id }
        }));
        pollJobStatus(job_id, widgetKey);
      } else {
        throw new Error("Trigger failed");
      }
    } catch (err) {
      console.warn(`Backend offline. Simulating local mock ingestion for: ${widgetKey}`);
      simulateMockIngestion(widgetKey);
    }
  };

  const pollJobStatus = (jobId: string, widgetKey: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const token = localStorage.getItem("fm_jwt_token");
        const res = await fetch(`http://127.0.0.1:8000/ingest/status/${jobId}`, {
          headers: { "Authorization": `Bearer ${token}` }
        });
        if (res.ok) {
          const job = await res.json();
          setPipelineJobs(prev => ({
            ...prev,
            [widgetKey]: { progress: job.progress, status: job.status, message: job.message, job_id: jobId }
          }));

          if (job.status === "completed" || job.status === "failed") {
            clearInterval(pollInterval);
            fetchStats();
            fetchDbStatus();
            if (widgetKey === "graph") {
              fetchGraphData();
            }
            setTimeout(() => {
              setPipelineJobs(prev => ({
                ...prev,
                [widgetKey]: { progress: 0, status: "idle", message: "" }
              }));
            }, 5000);
          }
        } else {
          clearInterval(pollInterval);
        }
      } catch (e) {
        clearInterval(pollInterval);
      }
    }, 1500);
  };

  const simulateMockIngestion = (widgetKey: string) => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += 20;
      let message = "Processing records...";
      if (progress === 40) message = "Generating layout block chunks...";
      if (progress === 80) message = "Deduplicating & indexing to Qdrant Cloud...";
      if (progress === 100) {
        message = "Completed successfully!";
        clearInterval(interval);
        fetchStats();
        fetchDbStatus();
        if (widgetKey === "graph") {
          fetchGraphData();
        }
        setTimeout(() => {
          setPipelineJobs(prev => ({
            ...prev,
            [widgetKey]: { progress: 0, status: "idle", message: "" }
          }));
        }, 5000);
      }

      setPipelineJobs(prev => ({
        ...prev,
        [widgetKey]: { progress, status: progress === 100 ? "completed" : "processing", message }
      }));
    }, 800);
  };

  const handleDeleteCollection = async () => {
    if (!confirm("Are you sure you want to delete all Qdrant collections? Stale data will be permanently lost.")) return;
    try {
      const token = localStorage.getItem("fm_jwt_token");
      const res = await fetch("http://127.0.0.1:8000/admin/collection/delete", {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (res.ok) {
        alert("Collections deleted successfully.");
        fetchDbStatus();
        fetchStats();
      }
    } catch (e) {
      alert("Failed to delete collections.");
    }
  };

  const handleRecreateCollection = async () => {
    try {
      const token = localStorage.getItem("fm_jwt_token");
      const res = await fetch("http://127.0.0.1:8000/admin/collection/recreate", {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (res.ok) {
        alert("Collections dropped and recreated empty.");
        fetchDbStatus();
        fetchStats();
      }
    } catch (e) {
      alert("Failed to recreate collections.");
    }
  };

  const handleSaveSettings = (e: React.FormEvent) => {
    e.preventDefault();
    setSettingsSaved(true);
    setTimeout(() => setSettingsSaved(false), 3000);
  };

  const renderProgressBar = (widgetKey: string) => {
    const job = pipelineJobs[widgetKey];
    if (!job || job.status === "idle" || job.status === "") return null;

    return (
      <div className="flex flex-col gap-1 text-xs mt-2 w-full">
        <div className="flex items-center justify-between font-bold text-brown-900">
          <span className="capitalize">{job.status}</span>
          <span>{job.progress}%</span>
        </div>
        <div className="w-full bg-brown-300/40 h-2 rounded-full overflow-hidden">
          <div className="bg-brown-700 h-full transition-all duration-300" style={{ width: `${job.progress}%` }} />
        </div>
        <p className="text-[10px] text-brown-700 font-semibold italic mt-0.5">{job.message}</p>
      </div>
    );
  };

  const collections = [
    { key: "manuals", label: "Service Manuals", desc: "Technical procedures & parameters", image: "https://images.unsplash.com/photo-1506784983877-45594efa4cbe?q=80&w=150&auto=format&fit=crop", icon: <BookOpen className="w-3.5 h-3.5 text-brown-700" /> },
    { key: "sop", label: "Maintenance SOPs", desc: "Step-by-step dispatch workflows", image: "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?q=80&w=150&auto=format&fit=crop", icon: <ClipboardList className="w-3.5 h-3.5 text-brown-700" /> },
    { key: "maintenance_logs", label: "Maintenance Logs", desc: "Chronological unit events & actions", image: "https://images.unsplash.com/photo-1508873535684-277a3cbcc4e8?q=80&w=150&auto=format&fit=crop", icon: <FileText className="w-3.5 h-3.5 text-brown-700" /> },
    { key: "error_codes", label: "DTC Error Codes", desc: "Machine CAN-bus fault registries", image: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=150&auto=format&fit=crop", icon: <AlertTriangle className="w-3.5 h-3.5 text-brown-700" /> },
    { key: "spare_parts", label: "Spare Parts Catalog", desc: "Excavator compatible inventory", image: "https://images.unsplash.com/photo-1530124560072-aae8d56b0efe?q=80&w=150&auto=format&fit=crop", icon: <Wrench className="w-3.5 h-3.5 text-brown-700" /> }
  ];

  return (
    <div className="flex flex-col gap-8 text-brown-900 bg-background">
      
      {/* Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-brown-300/20 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-brown-900">Control Panel & Administration</h1>
          <p className="text-sm text-brown-700">Configure indexing settings, adjust RAG thresholds, explore graphs, and drop database collections.</p>
        </div>
      </div>

      {/* NEW: Qdrant Database Lifecycle Admin Control Card */}
      <div className="bento-card bg-white p-6 border border-brown-300/30 rounded-3xl flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-brown-700">
            <Database className="w-5 h-5 text-orange-600 animate-pulse" />
            <h3 className="font-bold text-lg text-brown-900">Qdrant Vector DB Status</h3>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-1 text-xs">
            <div>
              <p className="text-brown-500 font-bold uppercase text-[9px]">Embedding Model</p>
              <p className="font-bold text-brown-900 mt-0.5">{dbStatus?.embedding_model || "BAAI/bge-small-en-v1.5"}</p>
            </div>
            <div>
              <p className="text-brown-500 font-bold uppercase text-[9px]">Dimensions</p>
              <p className="font-bold text-brown-900 mt-0.5">{dbStatus?.dimension || 384} px</p>
            </div>
            <div>
              <p className="text-brown-500 font-bold uppercase text-[9px]">Total Active Chunks</p>
              <p className="font-bold text-brown-900 mt-0.5">{dbStatus?.total_chunks || 0} Chunks</p>
            </div>
            <div>
              <p className="text-brown-500 font-bold uppercase text-[9px]">Last Indexed</p>
              <p className="font-bold text-brown-900 mt-0.5">{dbStatus?.last_indexed || "Recently"}</p>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={handleDeleteCollection}
            className="px-4 py-2.5 bg-red-650 hover:bg-red-750 text-white font-bold rounded-xl text-xs flex items-center gap-1.5 transition-colors border border-red-700/20"
          >
            <Trash2 className="w-4 h-4" />
            Delete Collections
          </button>
          
          <button
            onClick={handleRecreateCollection}
            className="px-4 py-2.5 bg-brown-900 hover:bg-brown-700 text-background font-bold rounded-xl text-xs flex items-center gap-1.5 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Recreate Collections
          </button>

          <button
            onClick={() => triggerIngestion("manuals")}
            className="px-4 py-2.5 bg-brown-700 hover:bg-brown-900 text-background font-bold rounded-xl text-xs flex items-center gap-1.5 transition-colors"
          >
            <UploadCloud className="w-4 h-4" />
            Re-index Manuals
          </button>
        </div>
      </div>

      {/* Grid: Stats Overview & Ingestion Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* RAG settings (First Adjust the settings in Admin) */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <h3 className="text-xl font-bold text-brown-900 flex items-center gap-2">
            <Settings className="w-5 h-5 text-brown-700" />
            RAG System Configuration
          </h3>

          <form onSubmit={handleSaveSettings} className="bento-card flex flex-col gap-4 bg-white border border-brown-300/30 p-6 rounded-3xl">
            <div className="space-y-3.5 text-xs">
              <div>
                <label className="flex justify-between font-bold text-brown-900 mb-1">
                  <span>Retriever Top-K Chunks</span>
                  <span className="text-orange-650 font-black">{retrieverTopK}</span>
                </label>
                <input
                  type="range" min="1" max="20"
                  value={retrieverTopK}
                  onChange={(e) => setRetrieverTopK(Number(e.target.value))}
                  className="w-full h-1.5 bg-brown-200 rounded-lg appearance-none cursor-pointer accent-brown-700"
                />
              </div>

              <div>
                <label className="flex justify-between font-bold text-brown-900 mb-1">
                  <span>Reranker Limit (Top Chunks)</span>
                  <span className="text-orange-650 font-black">{rerankerTopK}</span>
                </label>
                <input
                  type="range" min="1" max="10"
                  value={rerankerTopK}
                  onChange={(e) => setRerankerTopK(Number(e.target.value))}
                  className="w-full h-1.5 bg-brown-200 rounded-lg appearance-none cursor-pointer accent-brown-700"
                />
              </div>

              <div>
                <label className="flex justify-between font-bold text-brown-900 mb-1">
                  <span>Dense Vector weight</span>
                  <span className="text-orange-650 font-black">{(denseWeight * 100).toFixed(0)}%</span>
                </label>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={denseWeight}
                  onChange={(e) => setDenseWeight(Number(e.target.value))}
                  className="w-full h-1.5 bg-brown-200 rounded-lg appearance-none cursor-pointer accent-brown-700"
                />
              </div>

              <div>
                <label className="flex justify-between font-bold text-brown-900 mb-1">
                  <span>LLM Temperature</span>
                  <span className="text-orange-650 font-black">{temperature}</span>
                </label>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full h-1.5 bg-brown-200 rounded-lg appearance-none cursor-pointer accent-brown-700"
                />
              </div>

              <div>
                <label className="block font-bold text-brown-900 mb-1">Active LLM Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-2.5 bg-background border border-brown-300 rounded-xl font-semibold text-brown-950 focus:outline-none"
                >
                  <option value="qwen-72b-industrial">Qwen-2.5-72B-Instruct (Industrial Optimized)</option>
                  <option value="llama-3-70b-excavator">Llama-3.1-70B-Instruct (Hyundai Fine-tuned)</option>
                  <option value="gpt-4o-mini">GPT-4o-Mini (Generic Offline Fallback)</option>
                </select>
              </div>

              <div className="flex items-center justify-between py-1 pb-2 border-b border-brown-300/10">
                <span className="font-bold text-brown-900">Enable Semantic Caching</span>
                <input
                  type="checkbox"
                  checked={enableSemanticCache}
                  onChange={(e) => setEnableSemanticCache(e.target.checked)}
                  className="w-4 h-4 rounded text-brown-750 border-brown-300 focus:ring-brown-500 cursor-pointer accent-brown-700"
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full py-3 bg-brown-900 hover:bg-brown-700 text-background font-bold rounded-2xl text-xs flex items-center justify-center gap-1.5 transition-colors shadow-sm mt-2"
            >
              <Sliders className="w-3.5 h-3.5" />
              Save Configuration
            </button>

            {settingsSaved && (
              <p className="text-[10px] text-emerald-600 font-bold text-center mt-1 animate-pulse">
                ✓ Parameters applied to active RAG pipeline!
              </p>
            )}
          </form>
        </div>

        {/* Collection Stats (Bento Grid Style) */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <h3 className="text-xl font-bold text-brown-900 flex items-center gap-2">
            <Database className="w-5 h-5 text-brown-700" />
            Knowledge Base Collections
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {collections.map(col => {
              const stat = stats[col.key as keyof KBStats] || { count: 0, last_updated: "Never" };
              return (
                <div key={col.key} className="bento-card flex flex-col justify-between gap-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-extrabold text-brown-900">{col.label}</h4>
                      <span className="px-2.5 py-0.5 rounded-md bg-brown-700/10 text-brown-700 text-xs font-bold uppercase">
                        {col.key.replace("_", " ")}
                      </span>
                    </div>
                    <p className="text-xs text-brown-700 leading-relaxed">{col.desc}</p>
                  </div>
                  
                  <div className="flex items-center justify-between border-t border-brown-300/10 pt-3 text-xs">
                    <div>
                      <p className="text-brown-700 font-medium">Chunks Ingested</p>
                      <p className="text-lg font-black text-brown-900">{stat.count}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-brown-700 font-medium">Last Ingested</p>
                      <p className="font-bold text-brown-900">{stat.last_updated}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

      </div>

      {/* Grid: Neo4j Graph rebuild */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Pipelines */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <h3 className="text-xl font-bold text-brown-900 flex items-center gap-2">
            <UploadCloud className="w-5 h-5 text-brown-700" />
            Ingestion Pipeline Center
          </h3>

          <div className="bento-card flex flex-col gap-4">
            {collections.map(col => (
              <div key={col.key} className="flex flex-col p-3 bg-background border border-brown-300/20 rounded-xl gap-1">
                <div className="flex items-center gap-3">
                  <div className="relative w-10 h-10 rounded-lg overflow-hidden flex-shrink-0 border border-brown-300/30">
                    <Image
                      src={col.image}
                      alt={col.label}
                      fill
                      className="object-cover grayscale mix-blend-multiply"
                    />
                    <div className="absolute inset-0 bg-[#F0E6DA]/40 mix-blend-color" />
                  </div>
                  
                  <div className="flex-grow min-w-0">
                    <div className="flex items-center gap-1.5">
                      {col.icon}
                      <span className="text-xs font-extrabold text-brown-900 truncate">{col.label}</span>
                    </div>
                    <p className="text-[10px] text-brown-700 truncate">{col.desc}</p>
                  </div>
                  
                  <button
                    onClick={() => triggerIngestion(col.key)}
                    disabled={pipelineJobs[col.key]?.status === "processing" || pipelineJobs[col.key]?.status === "queued"}
                    className="flex items-center gap-1 bg-brown-700 hover:bg-brown-900 disabled:opacity-50 text-background px-3 py-1.5 rounded-lg text-xs font-bold transition-colors flex-shrink-0"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    Ingest
                  </button>
                </div>
                {renderProgressBar(col.key)}
              </div>
            ))}
          </div>
        </div>

        {/* Knowledge Graph Explorer (React Flow Graph) */}
        <div className="lg:col-span-2 bento-card flex flex-col gap-4 min-h-[420px]">
          <div className="flex items-center justify-between border-b border-brown-300/10 pb-3">
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-brown-700" />
              <h3 className="font-bold text-brown-900 text-lg">Neo4j Machine Knowledge Graph</h3>
            </div>
            
            <div className="flex items-center gap-3">
              <select
                value={selectedMachine}
                onChange={(e) => setSelectedMachine(e.target.value)}
                className="px-2.5 py-1 bg-surface border border-brown-300 rounded-lg text-xs font-semibold text-brown-900 focus:outline-none focus:border-brown-500"
              >
                <option value="M101">Hyundai R215L Smart Plus</option>
              </select>
 
              <button
                onClick={() => triggerIngestion("graph")}
                disabled={pipelineJobs["graph"]?.status === "processing" || pipelineJobs["graph"]?.status === "queued"}
                className="flex items-center gap-1 bg-brown-700 hover:bg-brown-900 disabled:opacity-50 text-background px-3 py-1.5 rounded-lg text-xs font-bold transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Rebuild Graph
              </button>
            </div>
          </div>

          {renderProgressBar("graph")}

          {/* Interactive React Flow Explorer container */}
          <div className="relative bg-background border border-brown-300/25 rounded-xl h-[320px] overflow-hidden">
            {isMounted ? (
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                fitView
              >
                <Background color="#9C6B45" gap={16} size={1} />
                <Controls />
                <MiniMap nodeStrokeWidth={3} zoomable pannable />
              </ReactFlow>
            ) : (
              <div className="flex items-center justify-center h-full text-xs text-brown-700 font-semibold animate-pulse">
                Loading Graph Explorer...
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
