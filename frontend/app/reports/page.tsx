"use client";

import React, { useState } from "react";
import { FileText, Download, Calendar, HardDrive, ShieldCheck, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";

interface Report {
  id: string;
  machine_id: string;
  date: string;
  title: string;
  status: string;
  size: string;
}

const INITIAL_REPORTS: Report[] = [
  {
    id: "REP-M101-087",
    machine_id: "M101",
    date: "2026-07-16",
    title: "Main Pump Rotational Vibration Dispatch Plan",
    status: "dispatched",
    size: "14 KB"
  },
  {
    id: "REP-M101-052",
    machine_id: "M101",
    date: "2026-05-12",
    title: "Pump Shaft Misalignment Inspection",
    status: "completed",
    size: "12 KB"
  },
  {
    id: "REP-M102-041",
    machine_id: "M102",
    date: "2026-04-10",
    title: "Boom Cylinder Hydraulic Seal Leakage Dispatch Plan",
    status: "completed",
    size: "15 KB"
  },
  {
    id: "REP-M101-032",
    machine_id: "M101",
    date: "2026-03-22",
    title: "Flywheel Flange Coupling Inspection Report",
    status: "completed",
    size: "11 KB"
  }
];

export default function ReportsPage() {
  const [reports] = useState<Report[]>(INITIAL_REPORTS);

  const handleDownload = (id: string) => {
    const token = localStorage.getItem("fm_jwt_token") || "";
    window.open(`http://localhost:8000/reports/${id}/pdf?token=${encodeURIComponent(token)}`, "_blank");
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-brown-300/20 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-brown-900">Maintenance Dispatch Reports</h1>
          <p className="text-sm text-brown-700">Access and download generated diagnostic dispatch plans and equipment audit reports.</p>
        </div>
      </div>

      {/* Reports List */}
      <motion.div
        initial="hidden"
        animate="show"
        variants={{
          hidden: {},
          show: { transition: { staggerChildren: 0.08 } }
        }}
        className="flex flex-col gap-4"
      >
        {reports.map((report) => (
          <motion.div
            key={report.id}
            variants={{
              hidden: { opacity: 0, y: 12 },
              show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut" } }
            }}
            whileHover={{ scale: 1.005, y: -2 }}
            className="bento-card flex flex-col md:flex-row items-start md:items-center justify-between gap-4 hover:border-brown-700 transition-all duration-300"
          >
            <div className="flex items-start gap-4">
              <div className="p-3 bg-brown-700/10 text-brown-700 rounded-xl mt-1 md:mt-0">
                <FileText className="w-6 h-6" />
              </div>
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-brown-500 bg-surface-alt px-2 py-0.5 rounded uppercase">
                    {report.machine_id}
                  </span>
                  <span className="text-xs text-brown-700 font-semibold flex items-center gap-1">
                    <Calendar className="w-3.5 h-3.5" />
                    {report.date}
                  </span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full capitalize ${
                    report.status === "dispatched" ? "bg-warning/10 text-warning" : "bg-success/10 text-success"
                  }`}>
                    {report.status}
                  </span>
                </div>
                <h3 className="text-lg font-bold text-brown-900 leading-snug">
                  {report.title}
                </h3>
                <p className="text-xs text-brown-700 flex items-center gap-1.5 mt-0.5">
                  <HardDrive className="w-3.5 h-3.5" />
                  Doc ID: {report.id} • Size: {report.size}
                </p>
              </div>
            </div>

            <button
              onClick={() => handleDownload(report.id)}
              className="w-full md:w-auto flex items-center justify-center gap-2 bg-brown-700 hover:bg-brown-900 text-background px-5 py-3 rounded-xl text-sm font-bold transition-colors shadow-sm"
            >
              <Download className="w-4 h-4" />
              Download PDF
            </button>
          </motion.div>
        ))}
      </motion.div>

      {/* Safety audit check */}
      <div className="bento-card bg-surface-alt/10 border-dashed flex items-start gap-3.5">
        <ShieldCheck className="w-6 h-6 text-success shrink-0" />
        <div className="text-xs leading-normal">
          <p className="font-extrabold text-brown-900 mb-1">Grounded PDF Reports Policy</p>
          <p className="text-brown-700">
            Every downloaded dispatch report compiles the active sensor measurements, manual citations, and components graph path captured by the supervisor agent during query execution, serving as a legally valid equipment maintenance record.
          </p>
        </div>
      </div>
    </div>
  );
}
