"use client";

import React, { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { 
  Send, Bot, User, RotateCcw, Trash2, Copy, ThumbsUp, ThumbsDown, StopCircle, 
  Mic, Paperclip, FileText, CheckCircle2, AlertTriangle, HelpCircle, 
  Thermometer, ShieldAlert, Navigation, ArrowRight, BookOpen, Layers, 
  Network, Activity, Sparkles, HardHat, File, Info, ExternalLink
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Citation {
  id: string;
  title: string;
  text: string;
  score: number;
  source_type: string;
  payload?: {
    image_path?: string;
    page?: number;
  };
}

interface Telemetry {
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
  vibration: number;
}

interface GraphEdge {
  source: string;
  relationship: string;
  target: string;
}

interface Evidence {
  citations: Citation[];
  sensor_values: Telemetry;
  kg_path: GraphEdge[];
  confidence_score: number;
  confidence_breakdown?: {
    overall: string;
    retrieval: string;
    graph: string;
    evidence: string;
    agreement: string;
  };
}

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: string;
  evidence?: Evidence;
  queryId?: string;
  feedback?: "like" | "dislike" | null;
}

const QUICK_ACTIONS = [
  { text: "Summarize uploaded manuals", query: "Summarize the key sections of the uploaded Hyundai R215L service manuals." },
  { text: "Explain hydraulic system", query: "Explain the structure and operation of the Hyundai R215L hydraulic system." },
  { text: "Find maintenance schedule", query: "What is the standard preventive maintenance schedule and intervals for the R215L excavator?" },
  { text: "Locate troubleshooting section", query: "Where in the troubleshooting manual is the oil pressure warning light diagnostic guide?" },
  { text: "Explain error code", query: "What are the common mechatronics error codes related to pump pressure sensors?" },
  { text: "Generate inspection checklist", query: "Generate a field inspection checklist for the bucket cylinders and mounting torque." },
  { text: "Show lubrication intervals", query: "What are the lubrication intervals and recommended greases for the swing circle and turntable bearing?" }
];

export default function AskPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const [isDebugOpen, setIsDebugOpen] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [userName, setUserName] = useState("Engineer");

  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedName = localStorage.getItem("fm_user_name");
      if (savedName) setUserName(savedName);
    }
  }, []);

  // Dynamic status/document stats
  const [stats, setStats] = useState({
    machine_model: "Hyundai R215L Smart Plus",
    manuals_count: 11,
    points_count: 0,
    pages_count: 845,
    tables_count: 148,
    images_count: 79
  });
  const [documents, setDocuments] = useState<any[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const loadingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const loadingSteps = [
    "Parsing question",
    "Searching manuals",
    "Finding diagrams",
    "Retrieving tables",
    "Ranking evidence",
    "Preparing answer",
    "Streaming response"
  ];

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Handle loading pipeline step intervals
  useEffect(() => {
    if (isLoading) {
      setLoadingStep(0);
      loadingIntervalRef.current = setInterval(() => {
        setLoadingStep((prev) => (prev < loadingSteps.length - 1 ? prev + 1 : prev));
      }, 900);
    } else {
      if (loadingIntervalRef.current) clearInterval(loadingIntervalRef.current);
    }
    return () => {
      if (loadingIntervalRef.current) clearInterval(loadingIntervalRef.current);
    };
  }, [isLoading]);

  // Load stats and documents on mount
  useEffect(() => {
    const token = typeof window !== "undefined" ? localStorage.getItem("fm_jwt_token") : null;
    const headers = token ? { "Authorization": `Bearer ${token}` } : {};

    fetch("http://127.0.0.1:8000/stats", { headers })
      .then(res => res.ok ? res.json() : null)
      .then(data => { if (data) setStats(data); })
      .catch(err => console.warn("Failed to fetch stats", err));

    fetch("http://127.0.0.1:8000/documents", { headers })
      .then(res => res.ok ? res.json() : null)
      .then(data => { if (data) setDocuments(data); })
      .catch(err => console.warn("Failed to fetch documents", err));
  }, [messages]);

  // Load chat history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("fm_chat_history");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setMessages(parsed);
        if (parsed.length > 0) {
          const lastAssistant = [...parsed].reverse().find(m => m.role === "assistant");
          if (lastAssistant) setActiveMessageId(lastAssistant.id);
        }
      } catch (e) {
        console.error("Failed to load chat history", e);
      }
    }
  }, []);

  const saveHistory = (newMessages: Message[]) => {
    setMessages(newMessages);
    localStorage.setItem("fm_chat_history", JSON.stringify(newMessages));
  };

  const handleAsk = async (customQuery?: string) => {
    const textToSend = customQuery || query;
    if (!textToSend.trim() || isLoading) return;

    if (!customQuery) setQuery("");

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg: Message = {
      id: Math.random().toString(36).substring(7),
      role: "user",
      text: textToSend,
      timestamp
    };

    const updatedMessages = [...messages, userMsg];
    saveHistory(updatedMessages);
    setIsLoading(true);

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const token = typeof window !== "undefined" ? localStorage.getItem("fm_jwt_token") : null;
      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          ...(token ? { "Authorization": `Bearer ${token}` } : {})
        },
        body: JSON.stringify({ query: textToSend, machine_id: "M101" }),
        signal: controller.signal
      });

      if (res.ok) {
        const data = await res.json();
        const assistantMsg: Message = {
          id: Math.random().toString(36).substring(7),
          role: "assistant",
          text: data.answer,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          evidence: data.evidence,
          queryId: data.query_id
        };
        const nextMsgs = [...updatedMessages, assistantMsg];
        saveHistory(nextMsgs);
        setActiveMessageId(assistantMsg.id);
      } else {
        throw new Error("Server returned an error");
      }
    } catch (err: any) {
      if (err.name === "AbortError") return;
      console.error("Backend request failed:", err);
      
      const assistantMsg: Message = {
        id: Math.random().toString(36).substring(7),
        role: "assistant",
        text: "⚠️ **Connection Error**: Unable to communicate with the FactoryMind AI backend RAG server. Please verify that the uvicorn backend service is running locally on `http://127.0.0.1:8000`.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      const nextMsgs = [...updatedMessages, assistantMsg];
      saveHistory(nextMsgs);
      setActiveMessageId(assistantMsg.id);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = () => {
    if (abortController) {
      abortController.abort();
      setIsLoading(false);
    }
  };

  const handleFeedback = (id: string, type: "like" | "dislike") => {
    const next = messages.map(m => {
      if (m.id === id) {
        return { ...m, feedback: m.feedback === type ? null : type };
      }
      return m;
    });
    saveHistory(next);
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const handleDownloadReport = (queryId: string) => {
    if (!queryId) return;
    window.open(`http://127.0.0.1:8000/reports/${queryId}/pdf`, "_blank");
  };

  // Custom lightweight Markdown renderer to remove ugly raw hash symbols
  const renderMarkdown = (text: string) => {
    const lines = text.split("\n");
    return lines.map((line, idx) => {
      const cleanLine = line.trim();
      
      // 1. Headings (e.g. ### Header)
      if (cleanLine.startsWith("###")) {
        return (
          <h3 key={idx} className="text-sm font-bold text-brown-900 mt-4 mb-2 pb-1 border-b border-brown-300/10">
            {cleanLine.replace("###", "").trim()}
          </h3>
        );
      }
      if (cleanLine.startsWith("##")) {
        return (
          <h2 key={idx} className="text-base font-black text-brown-900 mt-5 mb-2.5">
            {cleanLine.replace("##", "").trim()}
          </h2>
        );
      }
      if (cleanLine.startsWith("#")) {
        return (
          <h1 key={idx} className="text-lg font-black text-brown-900 mt-6 mb-3">
            {cleanLine.replace("#", "").trim()}
          </h1>
        );
      }

      // 2. Bold tags (e.g. **bold**)
      let parts = line.split("**");
      if (parts.length > 1) {
        return (
          <p key={idx} className="mb-2 leading-relaxed text-brown-900/80 text-xs md:text-sm">
            {parts.map((part, pIdx) => pIdx % 2 === 1 ? <strong key={pIdx} className="font-bold text-brown-900">{part}</strong> : part)}
          </p>
        );
      }

      // 3. Bullet lists
      if (cleanLine.startsWith("- ") || cleanLine.startsWith("* ")) {
        const content = cleanLine.replace(/^[-*]\s+/, "");
        return (
          <ul key={idx} className="list-disc pl-5 mb-1.5 text-brown-900/80 text-xs md:text-sm">
            <li>{content}</li>
          </ul>
        );
      }

      if (!cleanLine) {
        return <div key={idx} className="h-2" />;
      }

      return (
        <p key={idx} className="mb-2 leading-relaxed text-brown-900/85 text-xs md:text-sm">
          {line}
        </p>
      );
    });
  };

  const activeMessage = messages.find(m => m.id === activeMessageId);
  const activeEvidence = activeMessage?.evidence;

  // Confidence ratings breakdown
  const breakdown = activeEvidence?.confidence_breakdown || {
    overall: activeEvidence ? "High" : "Low",
    retrieval: activeEvidence ? "High" : "Low",
    graph: activeEvidence ? "High" : "Low",
    evidence: activeEvidence ? "High" : "Low",
    agreement: activeEvidence ? "High" : "Low"
  };

  return (
    <div className="flex flex-col lg:flex-row gap-6 h-[calc(100vh-8.5rem)] text-brown-900 bg-background">
      
      {/* 1. Left Sidebar: Dynamic Knowledge Base Details */}
      <div 
        data-lenis-prevent
        className="w-full lg:w-72 bg-white border border-brown-300/30 rounded-3xl p-5 flex flex-col gap-5 shadow-sm shrink-0 h-full overflow-y-auto custom-scrollbar"
      >
        <div>
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-brown-300">Knowledge Base</h2>
          <h3 className="text-sm font-extrabold text-brown-900 mt-1">{stats.machine_model}</h3>
        </div>

        {/* Dynamic Database Statistics */}
        <div className="space-y-2.5 bg-surface-alt/25 p-4 rounded-2xl border border-brown-300/10">
          <div className="flex justify-between text-xs">
            <span className="text-brown-700/60 font-medium">Manuals Indexed:</span>
            <span className="font-bold text-brown-900">{stats.manuals_count}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-brown-700/60 font-medium">Pages Scanned:</span>
            <span className="font-bold text-brown-900">{stats.pages_count}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-brown-700/60 font-medium">Tables Loaded:</span>
            <span className="font-bold text-brown-900">{stats.tables_count}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-brown-700/60 font-medium">Figures Extracted:</span>
            <span className="font-bold text-brown-900">{stats.images_count}</span>
          </div>
        </div>

        {/* Real Document List */}
        <div className="flex-1 flex flex-col gap-3 min-h-[220px]">
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-brown-500 flex items-center gap-1.5 pb-1 border-b border-brown-300/15">
            <FileText className="w-3.5 h-3.5" />
            Uploaded Documents ({documents.length || stats.manuals_count})
          </h4>

          <div 
            data-lenis-prevent
            className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar"
          >
            {documents.length > 0 ? (
              documents.map((doc, idx) => (
                <div key={idx} className="p-2.5 bg-white border border-brown-300/20 rounded-xl hover:border-brown-550 transition-all flex items-start gap-2 group">
                  <File className="w-3.5 h-3.5 text-brown-700 shrink-0 mt-0.5" />
                  <div className="flex-grow min-w-0">
                    <span className="text-[11px] font-semibold text-brown-900 block truncate group-hover:text-brown-700 transition-colors" title={doc.name}>
                      {doc.name}
                    </span>
                    <span className="text-[9px] text-brown-500 font-semibold block mt-0.5">
                      {doc.size_kb} KB • {doc.type.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className="space-y-2">
                <div className="p-2.5 bg-white border border-brown-300/20 rounded-xl flex items-start gap-2">
                  <File className="w-3.5 h-3.5 text-brown-700 shrink-0 mt-0.5" />
                  <span className="text-[11px] font-semibold text-brown-900 truncate">hyundai-r215l-smart-component-mounting-torque-manual.pdf</span>
                </div>
                <div className="p-2.5 bg-white border border-brown-300/20 rounded-xl flex items-start gap-2">
                  <File className="w-3.5 h-3.5 text-brown-700 shrink-0 mt-0.5" />
                  <span className="text-[11px] font-semibold text-brown-900 truncate">hyundai-r215l-smart-hydraulic-system-manual.pdf</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 2. Main Chat Panel (ChatGPT layout) */}
      <div className="flex-1 flex flex-col rounded-3xl bg-white border border-brown-300/30 overflow-hidden shadow-sm relative">
        
        {/* Top Header */}
        <div className="px-6 py-4 border-b border-brown-300/10 flex items-center justify-between z-10 bg-white/85 backdrop-blur-md">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-brown-700 flex items-center justify-center">
              <Bot className="w-5 h-5 text-background stroke-[2]" />
            </div>
            <div>
              <h2 className="font-extrabold text-sm text-brown-900 leading-none">FactoryMind AI</h2>
              <span className="text-[10px] text-brown-550 font-bold tracking-wide uppercase block mt-1">
                Copilot Session: {userName}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setDevMode(!devMode)}
              className={`px-2.5 py-1 rounded-lg border text-[10px] font-bold uppercase tracking-wide transition-all duration-200 ${
                devMode 
                  ? "bg-orange-600 border-orange-600 text-white shadow-sm font-black" 
                  : "bg-white border-brown-300/35 text-brown-750 hover:bg-slate-50 font-bold"
              }`}
            >
              Developer Mode: {devMode ? "ON" : "OFF"}
            </button>
            <div className="flex items-center gap-1.5 border-l border-brown-300/20 pl-3">
              <span className="w-2 h-2 rounded-full bg-emerald-600" />
              <span className="text-[10px] text-brown-700 font-bold uppercase tracking-wider">Ready</span>
            </div>
          </div>
        </div>

        {/* Message Thread */}
        <div 
          data-lenis-prevent
          className="flex-1 overflow-y-auto px-6 py-6 space-y-6 custom-scrollbar bg-surface-alt/10"
        >
          <AnimatePresence initial={false}>
            {messages.length === 0 ? (
              // Welcomer screen
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center py-8 text-center max-w-xl mx-auto h-full"
              >
                <div className="w-16 h-16 rounded-2xl bg-surface-alt border border-brown-300/20 flex items-center justify-center mb-5">
                  <Bot className="w-8 h-8 text-brown-700 stroke-[1.5]" />
                </div>
                <h1 className="text-2xl font-black text-brown-900 mb-1.5">Welcome back, {userName}!</h1>
                <p className="text-xs text-brown-700 max-w-sm mb-8 leading-relaxed font-semibold">
                  Ask technical questions about the Hyundai R215L Smart Plus excavator. Every answer is grounded directly in the indexed manuals.
                </p>

                <div className="w-full space-y-2">
                  {QUICK_ACTIONS.map((item, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleAsk(item.query)}
                      className="w-full p-3 bg-white hover:bg-surface-alt/20 border border-brown-300/30 hover:border-brown-300 rounded-xl transition-all flex items-center justify-between text-left group"
                    >
                      <span className="text-xs font-bold text-brown-900/80 group-hover:text-brown-700">{item.text}</span>
                      <ArrowRight className="w-3.5 h-3.5 text-brown-300 group-hover:text-brown-700 transition-colors" />
                    </button>
                  ))}
                </div>
              </motion.div>
            ) : (
              // Chat bubble list
              messages.map((msg) => (
                <div
                  key={msg.id}
                  onClick={() => msg.role === "assistant" && setActiveMessageId(msg.id)}
                  className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"} cursor-pointer group`}
                >
                  {msg.role === "assistant" && (
                    <div className={`w-8.5 h-8.5 rounded-lg flex items-center justify-center shrink-0 border transition-all ${
                      activeMessageId === msg.id 
                        ? "bg-brown-700 border-brown-500 text-background" 
                        : "bg-surface-alt/40 border-brown-300/30 text-brown-700"
                    }`}>
                      <Bot className="w-4.5 h-4.5 stroke-[2]" />
                    </div>
                  )}

                  <div className={`flex flex-col gap-1 max-w-[80%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                    <div className="flex items-center gap-1.5 text-[9px] text-brown-500 font-bold px-0.5">
                      <span>{msg.role === "user" ? "Maintenance Engineer" : "FactoryMind AI"}</span>
                      <span>•</span>
                      <span>{msg.timestamp}</span>
                    </div>

                    <div className={`p-4 rounded-xl leading-relaxed border shadow-sm ${
                      msg.role === "user"
                        ? "bg-brown-900 border-brown-900 text-white rounded-tr-none"
                        : `rounded-tl-none bg-white ${
                            activeMessageId === msg.id 
                              ? "border-brown-700 shadow-sm" 
                              : "border-brown-300/30 hover:border-brown-300"
                          }`
                    }`}>
                      {msg.role === "assistant" ? (
                        <div className="prose prose-stone max-w-none text-brown-900/90 whitespace-pre-wrap">
                          {renderMarkdown(msg.text)}
                        </div>
                      ) : (
                        <p className="text-xs md:text-sm font-medium">{msg.text}</p>
                      )}
                    </div>

                    {msg.role === "assistant" && (
                      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity pl-0.5 pt-0.5">
                        <button
                          onClick={() => handleCopy(msg.text)}
                          title="Copy answer"
                          className="p-1 hover:bg-surface-alt/30 rounded text-brown-500 hover:text-brown-700 transition-colors"
                        >
                          <Copy className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleFeedback(msg.id, "like")}
                          className={`p-1 hover:bg-surface-alt/30 rounded transition-colors ${msg.feedback === "like" ? "text-emerald-600" : "text-brown-500"}`}
                        >
                          <ThumbsUp className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleFeedback(msg.id, "dislike")}
                          className={`p-1 hover:bg-surface-alt/30 rounded transition-colors ${msg.feedback === "dislike" ? "text-red-500" : "text-brown-500"}`}
                        >
                          <ThumbsDown className="w-3.5 h-3.5" />
                        </button>
                        {msg.queryId && (
                          <button
                            onClick={() => handleDownloadReport(msg.queryId!)}
                            className="flex items-center gap-1 px-2.5 py-0.5 bg-surface-alt/30 hover:bg-surface-alt/60 border border-brown-300/30 rounded text-[10px] text-brown-700 hover:text-brown-900 transition-all font-bold ml-2"
                          >
                            <FileText className="w-3 h-3 text-orange-600" />
                            Download Report PDF
                          </button>
                        )}
                      </div>
                    )}
                  </div>

                  {msg.role === "user" && (
                    <div className="w-8.5 h-8.5 rounded-lg bg-surface-alt/40 border border-brown-300/30 text-brown-700 flex items-center justify-center shrink-0">
                      <User className="w-4.5 h-4.5" />
                    </div>
                  )}
                </div>
              ))
            )}

            {/* Pipeline progress loader steps */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-4 justify-start"
              >
                <div className="w-8.5 h-8.5 rounded-lg bg-surface-alt/40 border border-brown-300/20 flex items-center justify-center shrink-0 animate-pulse">
                  <Bot className="w-4.5 h-4.5 text-brown-500" />
                </div>
                <div className="flex flex-col gap-1.5 max-w-[80%]">
                  <div className="flex items-center gap-1.5 text-[9px] text-brown-500 font-bold px-0.5">
                    <span>RAG Pipeline</span>
                    <span>•</span>
                    <span className="text-orange-600 animate-pulse">Running</span>
                  </div>

                  <div className="rounded-xl rounded-tl-none border border-brown-300/30 bg-white p-5 space-y-4 shadow-sm min-w-[280px]">
                    <div className="flex items-center gap-2.5">
                      <div className="w-2 h-2 rounded-full bg-orange-600 animate-ping" />
                      <span className="text-[11px] text-brown-700 font-bold tracking-wide uppercase">
                        {loadingSteps[loadingStep]}...
                      </span>
                    </div>

                    <div className="space-y-1.5">
                      <div className="h-1.5 w-40 bg-surface-alt/30 rounded-full animate-pulse" />
                      <div className="h-1.5 w-52 bg-surface-alt/30 rounded-full animate-pulse" />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div ref={messagesEndRef} />
        </div>

        {/* Developer Mode Debug Panel */}
        {devMode && activeMessage && activeMessage.evidence && (
          <div className="border-t border-brown-300/25 bg-slate-50 p-5 space-y-4 text-xs z-15">
            <div className="flex items-center justify-between border-b border-brown-300/10 pb-2">
              <h4 className="font-extrabold text-brown-900 uppercase tracking-wider">
                Developer Debug Mode — Grounded Evidence
              </h4>
              <span className="text-[10px] text-brown-700 font-bold uppercase tracking-wider">
                {activeMessage.evidence.citations.length} Chunks Retrieved
              </span>
            </div>

            <div>
              <h5 className="font-bold text-brown-900 mb-1">Raw prompt sent to LLM:</h5>
              <pre className="p-3 bg-white border border-brown-300/20 rounded-xl font-mono text-[10px] text-brown-950 whitespace-pre-wrap max-h-32 overflow-y-auto custom-scrollbar">
                {activeMessage.evidence.llm_prompt}
              </pre>
            </div>

            <div>
              <h5 className="font-bold text-brown-900 mb-1.5">Retrieved Chunks & Similarity Scores:</h5>
              <div className="space-y-2.5 max-h-40 overflow-y-auto pr-1 custom-scrollbar">
                {activeMessage.evidence.citations.map((cite, cIdx) => (
                  <div key={cite.id} className="p-3 bg-white border border-brown-300/20 rounded-xl">
                    <div className="flex items-center justify-between border-b border-brown-300/10 pb-1 mb-1.5 text-[9px] font-bold text-brown-600">
                      <span>CHUNK #{cIdx + 1} ({cite.id})</span>
                      <div className="flex gap-2">
                        <span>Doc: {cite.title}</span>
                        {cite.payload?.page && <span>Page: {cite.payload.page}</span>}
                        <span className="text-emerald-700">RRF Score: {cite.score.toFixed(4)}</span>
                      </div>
                    </div>
                    <p className="text-brown-950/80 leading-relaxed italic">
                      "{cite.text}"
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h5 className="font-bold text-brown-900 mb-1">Final Answer Generated:</h5>
              <div className="p-3 bg-white border border-brown-300/20 rounded-xl text-brown-900 whitespace-pre-wrap">
                {activeMessage.text}
              </div>
            </div>
          </div>
        )}



        {/* Input Bar */}
        <div className="p-4 border-t border-brown-300/10 bg-white z-10">
          <form
            onSubmit={(e) => { e.preventDefault(); handleAsk(); }}
            className="relative flex items-center bg-surface-alt/25 border border-brown-300/30 focus-within:border-brown-700/50 rounded-2xl px-3 py-2 transition-all shadow-inner"
          >
            <button
              type="button"
              className="p-2 hover:bg-surface-alt/30 rounded-xl text-brown-500 hover:text-brown-700 transition-colors"
            >
              <Paperclip className="w-4.5 h-4.5" />
            </button>
            <button
              type="button"
              className="p-2 hover:bg-surface-alt/30 rounded-xl text-brown-500 hover:text-brown-700 transition-colors mr-1"
            >
              <Mic className="w-4.5 h-4.5" />
            </button>

            <textarea
              rows={1}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleAsk();
                }
              }}
              placeholder="Ask a technical question about the Hyundai R215L Smart Plus..."
              className="flex-grow bg-transparent text-brown-900 placeholder-brown-300/70 text-sm focus:outline-none resize-none max-h-24 py-1.5 pr-2.5"
            />

            {isLoading ? (
              <button
                type="button"
                onClick={handleStop}
                className="bg-orange-550 hover:bg-orange-600 text-white p-2.5 rounded-xl font-bold transition-all flex items-center justify-center"
              >
                <StopCircle className="w-4.5 h-4.5" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!query.trim()}
                className="bg-brown-700 hover:bg-brown-900 disabled:opacity-40 text-background p-2.5 rounded-xl font-bold transition-all flex items-center justify-center border border-brown-900/10"
              >
                <Send className="w-4.5 h-4.5" />
              </button>
            )}
          </form>
        </div>

      </div>

      {/* 3. Right Sidebar: Grounded Explainability Source Panel */}
      <div 
        data-lenis-prevent
        className="w-full lg:w-96 flex flex-col gap-5 shrink-0 h-full overflow-y-auto pr-1 custom-scrollbar"
      >
        
        {/* A. Retrieval Confidence Score */}
        <div className="rounded-3xl bg-white border border-brown-300/30 p-5 flex flex-col gap-4 shadow-sm">
          <div className="flex items-center gap-2 pb-2 border-b border-brown-300/10">
            <Sparkles className="w-4.5 h-4.5 text-brown-700" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-brown-550">
              Retrieval Confidence
            </h3>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <span className={`px-2.5 py-1 rounded-full text-xs font-bold tracking-wide uppercase ${
                breakdown.overall === "High" 
                  ? "bg-emerald-50 text-emerald-600 border border-emerald-100" 
                  : breakdown.overall === "Medium"
                  ? "bg-amber-50 text-amber-600 border border-amber-100"
                  : "bg-surface-alt/40 text-brown-500 border border-brown-300/20"
              }`}>
                {breakdown.overall} Confidence
              </span>
              <p className="text-[10px] text-brown-500 font-semibold mt-1.5">No fabricated information evaluated</p>
            </div>
            <div className={`w-3.5 h-3.5 rounded-full ${
              breakdown.overall === "High" 
                ? "bg-emerald-500" 
                : breakdown.overall === "Medium"
                ? "bg-amber-500"
                : "bg-brown-300"
            }`} />
          </div>

          <div className="space-y-2.5 pt-1 text-xs">
            <div className="flex justify-between">
              <span className="text-brown-700/60 font-medium">Retriever Similarity:</span>
              <span className="font-extrabold text-brown-900 uppercase text-[10px]">{breakdown.retrieval}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-brown-700/60 font-medium">Knowledge Graph Match:</span>
              <span className="font-extrabold text-brown-900 uppercase text-[10px]">{breakdown.graph}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-brown-700/60 font-medium">Evidence Coverage:</span>
              <span className="font-extrabold text-brown-900 uppercase text-[10px]">{breakdown.evidence}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-brown-700/60 font-medium">Document Agreement:</span>
              <span className="font-extrabold text-brown-900 uppercase text-[10px]">{breakdown.agreement}</span>
            </div>
          </div>
        </div>

        {/* B. Live Telemetry Panel */}
        <div className="rounded-3xl bg-white border border-brown-300/30 p-5 flex flex-col gap-4 shadow-sm">
          <div className="flex items-center gap-2 pb-2 border-b border-brown-300/10">
            <Activity className="w-4.5 h-4.5 text-brown-700" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-brown-550">
              Machinery Telemetry Feed
            </h3>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-surface-alt/10 p-3 rounded-xl border border-brown-300/10">
              <span className="text-[10px] font-bold text-brown-500 uppercase tracking-wide">Vibration</span>
              <span className="text-sm font-extrabold text-brown-900 mt-0.5 block">
                {activeEvidence?.sensor_values.vibration || "0.08"} mm
              </span>
            </div>
            <div className="bg-surface-alt/10 p-3 rounded-xl border border-brown-300/10">
              <span className="text-[10px] font-bold text-brown-500 uppercase tracking-wide">Torque</span>
              <span className="text-sm font-extrabold text-brown-900 mt-0.5 block">
                {activeEvidence?.sensor_values.torque || "45.2"} Nm
              </span>
            </div>
            <div className="bg-surface-alt/10 p-3 rounded-xl border border-brown-300/10">
              <span className="text-[10px] font-bold text-brown-500 uppercase tracking-wide">Speed</span>
              <span className="text-sm font-extrabold text-brown-900 mt-0.5 block">
                {activeEvidence?.sensor_values.rotational_speed || "1850"} RPM
              </span>
            </div>
            <div className="bg-surface-alt/10 p-3 rounded-xl border border-brown-300/10">
              <span className="text-[10px] font-bold text-brown-500 uppercase tracking-wide">Tool Wear</span>
              <span className="text-sm font-extrabold text-brown-900 mt-0.5 block">
                {activeEvidence?.sensor_values.tool_wear || "120"} min
              </span>
            </div>
          </div>
        </div>

        {/* C. Source Panel: References, page numbers and PDF links */}
        <div className="rounded-3xl bg-white border border-brown-300/30 p-5 flex flex-col gap-4 shadow-sm">
          <div className="flex items-center gap-2 pb-2 border-b border-brown-300/10">
            <BookOpen className="w-4.5 h-4.5 text-brown-700" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-brown-550">
              Manual Sources ({activeEvidence?.citations.length || 0})
            </h3>
          </div>

          <div 
            data-lenis-prevent
            className="space-y-4"
          >
            {activeEvidence?.citations && activeEvidence.citations.length > 0 ? (
              activeEvidence.citations.map((cite) => (
                <div key={cite.id} className="p-3 bg-surface-alt/10 border border-brown-300/15 rounded-2xl flex flex-col gap-2">
                  <div className="flex items-start justify-between gap-2 border-b border-brown-300/10 pb-1.5">
                    <div>
                      <h4 className="text-xs font-bold text-brown-900 truncate max-w-[190px]" title={cite.title}>
                        {cite.title}
                      </h4>
                      {cite.payload?.page && (
                        <span className="text-[10px] text-brown-500 font-semibold block mt-0.5">
                          Section reference • Page {cite.payload.page}
                        </span>
                      )}
                    </div>
                    <span className="text-[9px] px-2 py-0.5 bg-surface-alt border border-brown-300/20 text-brown-700 font-bold rounded-lg shrink-0">
                      RRF: {cite.score.toFixed(3)}
                    </span>
                  </div>

                  <p className="text-xs text-brown-900/80 leading-relaxed italic bg-white/70 p-2.5 rounded-xl border border-brown-300/10">
                    "...{cite.text}..."
                  </p>

                  <div className="flex items-center gap-2 mt-1">
                    <button
                      onClick={() => alert(`Expanding context block:\n${cite.text}`)}
                      className="px-2 py-1 bg-white hover:bg-surface-alt/25 border border-brown-300/20 rounded-lg text-[10px] font-bold text-brown-700 transition-colors"
                    >
                      Expand Context
                    </button>
                    {cite.payload?.image_path && (
                      <button
                        onClick={() => {
                          const w = window.open("", "_blank");
                          if (w) w.document.write(`<img src="${cite.payload?.image_path}" style="max-width:100%;height:auto;display:block;margin:20px auto;"/>`);
                        }}
                        className="px-2 py-1 bg-white hover:bg-surface-alt/25 border border-brown-300/20 rounded-lg text-[10px] font-bold text-brown-700 transition-colors inline-flex items-center gap-1"
                      >
                        Show Image
                        <ExternalLink className="w-2.5 h-2.5" />
                      </button>
                    )}
                  </div>

                  {cite.payload?.image_path && (
                    <div className="relative w-full h-44 mt-1 rounded-xl overflow-hidden border border-brown-300/20 bg-white flex items-center justify-center p-2">
                      <Image
                        src={cite.payload.image_path}
                        alt="Schematic excerpt"
                        fill
                        className="object-contain"
                      />
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-xs text-brown-400 italic border border-dashed border-brown-300/20 rounded-2xl bg-surface-alt/5">
                Grounded sources and diagrams will display here upon query submission.
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
}
