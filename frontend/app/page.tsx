"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { 
  Database, Compass, Cpu, ArrowUpRight, Bot, 
  FileText, Sparkles, File
} from "lucide-react";

export default function LandingPage() {
  const [stats, setStats] = useState({
    machine_model: "Hyundai R215L Smart Plus",
    manuals_count: 11,
    points_count: 0,
    pages_count: 845,
    tables_count: 148,
    images_count: 79
  });
  const [documents, setDocuments] = useState<any[]>([]);

  useEffect(() => {
    // Fetch stats
    fetch("http://127.0.0.1:8000/stats")
      .then(res => res.ok ? res.json() : null)
      .then(data => { if (data) setStats(data); })
      .catch(err => console.warn("Failed to fetch stats", err));

    // Fetch documents
    fetch("http://127.0.0.1:8000/documents")
      .then(res => res.ok ? res.json() : null)
      .then(data => { if (data) setDocuments(data); })
      .catch(err => console.warn("Failed to fetch documents", err));
  }, []);

  const quickActions = [
    {
      title: "Summarize uploaded manuals",
      query: "Summarize the key sections of the uploaded Hyundai R215L service manuals.",
      desc: "Get a high-level summary of the entire knowledge base"
    },
    {
      title: "Explain hydraulic system",
      query: "Explain the structure and operation of the Hyundai R215L hydraulic system.",
      desc: "Traces pump outputs, valve controls, and accumulator limits"
    },
    {
      title: "Find maintenance schedule",
      query: "What is the standard preventive maintenance schedule and intervals for the R215L excavator?",
      desc: "Check hourly intervals and inspection milestones"
    },
    {
      title: "Locate troubleshooting section",
      query: "Where in the troubleshooting manual is the oil pressure warning light diagnostic guide?",
      desc: "Direct navigation to pressure sensor warnings"
    },
    {
      title: "Explain error code",
      query: "What are the common mechatronics error codes related to pump pressure sensors?",
      desc: "Decode mechatronics faults and electrical system alerts"
    },
    {
      title: "Generate inspection checklist",
      query: "Generate a field inspection checklist for the bucket cylinders and mounting torque.",
      desc: "Downloadable safety check steps"
    },
    {
      title: "Show lubrication intervals",
      query: "What are the lubrication intervals and recommended greases for the swing circle and turntable bearing?",
      desc: "Lubricant specs and grease torque requirements"
    }
  ];

  return (
    <div className="flex flex-col gap-12 py-4 text-brown-900 bg-background max-w-6xl mx-auto">
      
      {/* Warm Clean Enterprise Hero Section */}
      <section className="relative rounded-3xl bg-surface border border-brown-300/30 overflow-hidden shadow-sm p-8 md:p-12 lg:p-14 flex flex-col lg:flex-row items-center gap-12">
        <div className="absolute inset-0 bg-gradient-to-br from-brown-500/5 via-transparent to-transparent z-0" />
        
        {/* Left Column */}
        <div className="flex-1 space-y-6 relative z-10 text-left">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-alt/40 border border-brown-300/30 text-xs font-semibold text-brown-700">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Hyundai R215L Smart Plus Excavator Copilot</span>
          </div>

          <h1 className="text-4xl md:text-5xl font-black text-brown-900 tracking-tight leading-none">
            FactoryMind <span className="text-brown-500 font-normal">AI</span>
          </h1>

          <h2 className="text-lg md:text-xl font-bold text-brown-700 leading-normal max-w-xl">
            Industrial Knowledge Copilot powered by Layout-Aware Agentic RAG.
          </h2>

          <p className="text-sm text-brown-900/70 max-w-lg leading-relaxed">
            Verify schematics, lookup mounting torques, and check troubleshooting checklists. FactoryMind AI extracts facts directly from indexed manuals with zero hallucination.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link
              href="/ask"
              className="px-6 py-3 bg-brown-700 hover:bg-brown-900 text-background font-bold rounded-2xl transition-all duration-300 shadow-sm flex items-center gap-2 group border border-brown-900/10"
            >
              Launch Copilot Chat
              <ArrowUpRight className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
            </Link>
            <Link
              href="/history"
              className="px-6 py-3 bg-surface hover:bg-surface-alt/40 border border-brown-300 text-brown-700 font-bold rounded-2xl transition-all duration-300 shadow-sm"
            >
              View Telemetry Feeds
            </Link>
          </div>
        </div>

        {/* Right Column: Premium Warm SVG Robot Illustration */}
        <div className="flex-grow flex justify-center items-center relative z-10 w-full max-w-xs md:max-w-sm">
          <div className="absolute inset-0 bg-brown-500/5 blur-3xl rounded-full" />
          <svg className="w-56 h-56 md:w-64 md:h-64 drop-shadow-md" viewBox="0 0 200 200" fill="none">
            {/* Outer Grid */}
            <circle cx="100" cy="100" r="85" stroke="#f0e6da" strokeWidth="1" />
            <circle cx="100" cy="100" r="70" stroke="#c9a87c" strokeWidth="0.8" strokeDasharray="3 3" strokeOpacity="0.4" />
            
            {/* Base */}
            <path d="M 60 170 L 140 170 L 125 140 L 75 140 Z" fill="#f0e6da" stroke="#c9a87c" strokeWidth="1.2" />
            <rect x="95" y="110" width="10" height="30" fill="#c9a87c" />
            
            {/* Head shell */}
            <rect x="55" y="45" width="90" height="65" rx="18" fill="#ffffff" stroke="#c9a87c" strokeWidth="2.2" />
            
            {/* Screen Face */}
            <rect x="64" y="54" width="72" height="47" rx="10" fill="#f0e6da" stroke="#9c6b45" strokeWidth="1.5" />
            <rect x="66" y="56" width="68" height="43" rx="8" fill="#ffffff" />
            
            {/* Eyes */}
            <circle cx="86" cy="76" r="6" fill="#ffffff" stroke="#9c6b45" strokeWidth="1.2" />
            <circle cx="86" cy="76" r="3" fill="#6b4a32" />
            
            <circle cx="114" cy="76" r="6" fill="#ffffff" stroke="#9c6b45" strokeWidth="1.2" />
            <circle cx="114" cy="76" r="3" fill="#6b4a32" />

            {/* Radars */}
            <line x1="100" y1="45" x2="100" y2="28" stroke="#9c6b45" strokeWidth="2" />
            <circle cx="100" cy="26" r="3.5" fill="#9c6b45" />
            
            {/* Antennas */}
            <rect x="47" y="65" width="8" height="25" rx="3" fill="#f0e6da" stroke="#c9a87c" strokeWidth="1" />
            <rect x="145" y="65" width="8" height="25" rx="3" fill="#f0e6da" stroke="#c9a87c" strokeWidth="1" />
          </svg>
        </div>
      </section>

      {/* Database / Knowledge Base Status Sidebar Block */}
      <section className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Stats Column */}
        <div className="p-6 bg-surface border border-brown-300/30 rounded-3xl flex flex-col gap-4 shadow-sm text-left">
          <div>
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-brown-500">Current Knowledge Base</h3>
            <h4 className="text-sm font-extrabold text-brown-900 mt-1">{stats.machine_model}</h4>
          </div>
          
          <div className="space-y-2 pt-2 text-xs">
            <div className="flex justify-between border-b border-brown-300/10 pb-1.5">
              <span className="text-brown-700/60 font-medium">Manuals:</span>
              <span className="font-extrabold text-brown-900">{stats.manuals_count} Indexed</span>
            </div>
            <div className="flex justify-between border-b border-brown-300/10 pb-1.5">
              <span className="text-brown-700/60 font-medium">Pages:</span>
              <span className="font-extrabold text-brown-900">{stats.pages_count} Pages</span>
            </div>
            <div className="flex justify-between border-b border-brown-300/10 pb-1.5">
              <span className="text-brown-700/60 font-medium">Tables:</span>
              <span className="font-extrabold text-brown-900">{stats.tables_count} Tables</span>
            </div>
            <div className="flex justify-between pb-1">
              <span className="text-brown-700/60 font-medium">Images:</span>
              <span className="font-extrabold text-brown-900">{stats.images_count} Diagrams</span>
            </div>
          </div>
        </div>

        {/* Real Manual list */}
        <div className="lg:col-span-3 p-6 bg-surface border border-brown-300/30 rounded-3xl shadow-sm text-left flex flex-col gap-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-brown-700 border-b border-brown-300/10 pb-2 flex items-center gap-1.5">
            <FileText className="w-4 h-4 text-brown-700" />
            Currently Indexed Manual Files ({documents.length || stats.manuals_count})
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 overflow-y-auto max-h-[160px] pr-1 custom-scrollbar">
            {documents.length > 0 ? (
              documents.map((doc, idx) => (
                <div key={idx} className="p-3 bg-surface-alt/25 border border-brown-300/20 rounded-2xl flex items-start gap-2.5">
                  <File className="w-4.5 h-4.5 text-brown-700 shrink-0 mt-0.5" />
                  <div className="min-w-0">
                    <span className="text-xs font-bold text-brown-900 block truncate" title={doc.name}>
                      {doc.name}
                    </span>
                    <span className="text-[10px] text-brown-500 font-semibold block mt-0.5">
                      {doc.size_kb} KB • {doc.type.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              // Backup default list matching seeded files
              <div className="p-3 bg-surface-alt/25 border border-brown-300/20 rounded-2xl flex items-start gap-2.5">
                <File className="w-4.5 h-4.5 text-brown-700 shrink-0 mt-0.5" />
                <div>
                  <span className="text-xs font-bold text-brown-900">hyundai-r215l-smart-hydraulic-system-manual.pdf</span>
                  <span className="text-[10px] text-brown-500 font-semibold block mt-0.5">2418 KB • PDF</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Quick Action Suggestion Cards */}
      <section className="space-y-4">
        <div className="text-left">
          <h2 className="text-xs font-extrabold uppercase tracking-wider text-brown-700">
            Suggested Dispatch Scenarios
          </h2>
          <p className="text-[11px] text-brown-500 mt-0.5">
            Select a quick action card to open the assistant and search grounded manuals.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {quickActions.slice(0, 4).map((action, idx) => (
            <Link
              key={idx}
              href={`/ask?q=${encodeURIComponent(action.query)}`}
              className="p-5 bg-surface hover:bg-surface-alt/30 border border-brown-350 hover:border-brown-700 rounded-2xl transition-all duration-300 flex flex-col justify-between min-h-[140px] text-left shadow-sm group"
            >
              <div className="p-2 bg-surface-alt/40 border border-brown-300/20 rounded-xl w-fit">
                <Bot className="w-4.5 h-4.5 text-brown-700" />
              </div>
              <div className="space-y-1 mt-6">
                <h4 className="text-xs font-bold text-brown-900 group-hover:text-brown-700 transition-colors">
                  {action.title}
                </h4>
                <p className="text-[10px] text-brown-500 line-clamp-1 leading-normal">
                  {action.desc}
                </p>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Feature capabilities grid */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-6 bg-surface border border-brown-300/30 rounded-3xl flex flex-col gap-4 text-left shadow-sm">
          <div className="p-2.5 bg-surface-alt/40 border border-brown-300/20 rounded-xl w-fit">
            <Database className="w-5.5 h-5.5 text-brown-700" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-brown-900 mb-1">Layout-Aware chunking</h3>
            <p className="text-[11px] text-brown-500 leading-relaxed">
              Docling structure preservation ensures headings, lists, tables, and schematics retain correct parent-child relationships.
            </p>
          </div>
        </div>
        
        <div className="p-6 bg-surface border border-brown-300/30 rounded-3xl flex flex-col gap-4 text-left shadow-sm">
          <div className="p-2.5 bg-surface-alt/40 border border-brown-300/20 rounded-xl w-fit">
            <Compass className="w-5.5 h-5.5 text-brown-700" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-brown-900 mb-1">Retrieval Grounding</h3>
            <p className="text-[11px] text-brown-500 leading-relaxed">
              Bypasses model memory to fetch exclusively from uploaded documents, raising warnings if query is outside manual coverage.
            </p>
          </div>
        </div>

        <div className="p-6 bg-surface border border-brown-300/30 rounded-3xl flex flex-col gap-4 text-left shadow-sm">
          <div className="p-2.5 bg-surface-alt/40 border border-brown-300/20 rounded-xl w-fit">
            <Cpu className="w-5.5 h-5.5 text-brown-700" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-brown-900 mb-1">Decoupled IoT Service</h3>
            <p className="text-[11px] text-brown-500 leading-relaxed">
              Predictive models mapped under clean DTO, repository, and service classes ready for timeseries telemetry stream integration.
            </p>
          </div>
        </div>
      </section>

    </div>
  );
}
