"use client";

import React, { useState } from "react";
import { Play, Sparkles, CheckCircle2, AlertTriangle, FileText, Database, ShieldAlert } from "lucide-react";

interface EvalResult {
  question: string;
  expectedSource: string;
  expectedPages: string;
  retrievedChunks: any[];
  llmAnswer: string;
  loading: boolean;
  status: "idle" | "loading" | "success" | "error";
}

export default function EvalPage() {
  const [evals, setEvals] = useState<EvalResult[]>([
    {
      question: "What oil should be used for the engine?",
      expectedSource: "hyundai-r215l-smart-general-manual.pdf",
      expectedPages: "Page 22 (Recommended Lubrication oil specifications)",
      retrievedChunks: [],
      llmAnswer: "",
      loading: false,
      status: "idle"
    },
    {
      question: "What is the engine oil capacity?",
      expectedSource: "hyundai-r215l-smart-maintenance-standard-manual.pdf / general-manual.pdf",
      expectedPages: "Refill capacities section",
      retrievedChunks: [],
      llmAnswer: "",
      loading: false,
      status: "idle"
    },
    {
      question: "What are the ambient temperature ranges?",
      expectedSource: "hyundai-r215l-smart-general-manual.pdf",
      expectedPages: "Working temperature limits table",
      retrievedChunks: [],
      llmAnswer: "",
      loading: false,
      status: "idle"
    },
    {
      question: "What is the pilot relief valve pressure limit?",
      expectedSource: "hyundai-r215l-smart-hydraulic-system-manual.pdf",
      expectedPages: "Section 3: Valve limits (Page 14 / 15)",
      retrievedChunks: [],
      llmAnswer: "",
      loading: false,
      status: "idle"
    }
  ]);

  const runSingleEval = async (index: number) => {
    const item = evals[index];
    setEvals(prev => {
      const next = [...prev];
      next[index] = { ...next[index], status: "loading", loading: true };
      return next;
    });

    try {
      const token = typeof window !== "undefined" ? localStorage.getItem("fm_jwt_token") : null;
      const headers = {
        "Content-Type": "application/json",
        ...(token ? { "Authorization": `Bearer ${token}` } : {})
      };

      // 1. Fetch retrieved chunks
      const retrieveRes = await fetch("http://127.0.0.1:8000/debug/retrieve", {
        method: "POST",
        headers,
        body: JSON.stringify({ query: item.question, top_k: 5 })
      });

      let chunks = [];
      if (retrieveRes.ok) {
        const retrieveData = await retrieveRes.json();
        chunks = retrieveData.chunks;
      }

      // 2. Fetch LLM Grounded Answer
      const queryRes = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers,
        body: JSON.stringify({ query: item.question, machine_id: "M101" })
      });

      let answer = "Error retrieving final answer.";
      if (queryRes.ok) {
        const queryData = await queryRes.json();
        answer = queryData.answer;
      }

      setEvals(prev => {
        const next = [...prev];
        next[index] = {
          ...next[index],
          retrievedChunks: chunks,
          llmAnswer: answer,
          loading: false,
          status: "success"
        };
        return next;
      });
    } catch (err) {
      setEvals(prev => {
        const next = [...prev];
        next[index] = {
          ...next[index],
          llmAnswer: "Failed to connect to backend RAG API.",
          loading: false,
          status: "error"
        };
        return next;
      });
    }
  };

  const runAllEvals = async () => {
    for (let i = 0; i < evals.length; i++) {
      await runSingleEval(i);
    }
  };

  return (
    <div className="flex flex-col gap-8 text-brown-900 bg-background max-w-6xl mx-auto">
      
      {/* Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-brown-300/20 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-brown-900">RAG Evaluation Benchmark</h1>
          <p className="text-sm text-brown-700">Validate retrieval quality, cross-reference similarity scores, and review LLM grounding accuracy.</p>
        </div>
        <button
          onClick={runAllEvals}
          className="flex items-center gap-1.5 bg-brown-750 hover:bg-brown-900 text-background px-6 py-3 rounded-2xl text-xs font-bold transition-all shadow-sm"
        >
          <Play className="w-4 h-4 fill-current" />
          Run All Benchmarks
        </button>
      </div>

      {/* Benchmark list */}
      <div className="space-y-6">
        {evals.map((item, idx) => (
          <div key={idx} className="bg-white border border-brown-300/30 rounded-3xl p-6 shadow-sm flex flex-col gap-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-brown-300/10 pb-3">
              <div className="flex items-center gap-2.5">
                <span className="w-5 h-5 rounded-full bg-brown-700 text-background text-[10px] font-bold flex items-center justify-center">
                  {idx + 1}
                </span>
                <h3 className="font-extrabold text-sm text-brown-900">{item.question}</h3>
              </div>

              <div className="flex items-center gap-3">
                <span className={`px-2.5 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${
                  item.status === "success" 
                    ? "bg-emerald-50 text-emerald-600 border border-emerald-100" 
                    : item.status === "loading"
                    ? "bg-orange-50 text-orange-600 border border-orange-100 animate-pulse"
                    : "bg-surface-alt text-brown-500"
                }`}>
                  {item.status}
                </span>
                
                <button
                  onClick={() => runSingleEval(idx)}
                  disabled={item.loading}
                  className="px-3 py-1.5 bg-surface-alt/45 hover:bg-surface-alt text-brown-700 hover:text-brown-900 disabled:opacity-50 text-xs font-bold rounded-lg transition-colors border border-brown-300/20"
                >
                  Run Query
                </button>
              </div>
            </div>

            {/* Expected vs Actual */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
              
              {/* Expected side */}
              <div className="space-y-3 p-4 bg-surface-alt/10 border border-brown-300/15 rounded-2xl">
                <div>
                  <h4 className="font-extrabold text-brown-700 uppercase text-[9px] tracking-wider mb-1">Expected Source manual</h4>
                  <p className="font-bold text-brown-900 flex items-center gap-1.5">
                    <FileText className="w-4 h-4 text-brown-700" />
                    {item.expectedSource}
                  </p>
                </div>
                <div>
                  <h4 className="font-extrabold text-brown-700 uppercase text-[9px] tracking-wider mb-1">Target Pages</h4>
                  <p className="font-semibold text-brown-900">{item.expectedPages}</p>
                </div>
              </div>

              {/* LLM Answer side */}
              <div className="space-y-2 p-4 bg-slate-50 border border-slate-100 rounded-2xl">
                <h4 className="font-extrabold text-brown-700 uppercase text-[9px] tracking-wider">Actual LLM Generated Answer</h4>
                {item.status === "loading" ? (
                  <div className="space-y-2 py-2">
                    <div className="h-3 w-40 bg-slate-200 rounded animate-pulse" />
                    <div className="h-3 w-full bg-slate-200 rounded animate-pulse" />
                  </div>
                ) : item.llmAnswer ? (
                  <p className="text-slate-700 leading-relaxed font-medium whitespace-pre-wrap">
                    {item.llmAnswer}
                  </p>
                ) : (
                  <p className="text-slate-400 italic">No benchmark ran yet. Click Run Query to invoke RAG.</p>
                )}
              </div>
            </div>

            {/* Retrieved Chunks list */}
            {item.retrievedChunks.length > 0 && (
              <div className="border-t border-brown-300/10 pt-4">
                <h4 className="text-xs font-bold text-brown-900 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                  <Database className="w-4 h-4 text-orange-600" />
                  Retrieved Chunks ({item.retrievedChunks.length})
                </h4>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {item.retrievedChunks.map((chunk, cIdx) => (
                    <div key={chunk.id} className="p-3.5 bg-slate-50 border border-slate-150 rounded-2xl text-[11px] flex flex-col justify-between min-h-[120px]">
                      <div>
                        <div className="flex items-center justify-between mb-1.5 border-b border-slate-200/40 pb-1.5">
                          <span className="font-extrabold text-[9px] text-brown-700 uppercase">
                            Chunk #{cIdx + 1}
                          </span>
                          <span className="bg-emerald-50 text-emerald-600 px-2 py-0.5 rounded font-extrabold text-[9px]">
                            Score: {chunk.score.toFixed(4)}
                          </span>
                        </div>
                        <p className="text-slate-600 italic leading-relaxed line-clamp-3">
                          "{chunk.text}"
                        </p>
                      </div>

                      <div className="mt-3 flex justify-between text-[10px] text-slate-500 font-bold border-t border-slate-200/20 pt-2">
                        <span className="truncate max-w-[170px]" title={chunk.document_name}>{chunk.document_name}</span>
                        <span>Page {chunk.page} • {chunk.heading}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
          </div>
        ))}
      </div>

    </div>
  );
}
