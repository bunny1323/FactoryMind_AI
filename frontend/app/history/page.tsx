"use client";

import React, { useState, useEffect } from "react";
import { Clock, CheckCircle, Wrench, ShieldAlert, TrendingUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";

interface LogEntry {
  date: string;
  issue: string;
  action: string;
  downtime: number;
  prob: number;
}

const MOCK_DATA: Record<string, { logs: LogEntry[]; chartData: any[] }> = {
  M101: {
    logs: [
      { date: "2026-01-15", issue: "Hydraulic pump temperature high (358 K)", action: "Cleaned hydraulic radiator and replaced hydraulic oil filters", downtime: 3.5, prob: 0.12 },
      { date: "2026-03-22", issue: "Minor hydraulic pump casing vibration (0.11 mm)", action: "Inspected coupling and tightened engine-pump flange mounting bolts", downtime: 2.0, prob: 0.25 },
      { date: "2026-05-12", issue: "Vibration alarm E-VIB01 triggered on main pump", action: "Inspected mounting brackets and aligned pump shaft to engine flywheel", downtime: 5.0, prob: 0.72 }
    ],
    chartData: [
      { month: "Jan", vibration: 0.04, prob: 12, temp: 72 },
      { month: "Feb", vibration: 0.08, prob: 18, temp: 74 },
      { month: "Mar", vibration: 0.11, prob: 25, temp: 77 },
      { month: "Apr", vibration: 0.06, prob: 14, temp: 73 },
      { month: "May", vibration: 0.22, prob: 72, temp: 85 },
      { month: "Jun", vibration: 0.05, prob: 10, temp: 70 },
      { month: "Jul", vibration: 0.09, prob: 19, temp: 73 }
    ]
  },
  M102: {
    logs: [
      { date: "2026-04-10", issue: "Boom cylinder seal leak", action: "Replaced boom cylinder seal kit and refilled hydraulic fluid", downtime: 4.0, prob: 0.05 }
    ],
    chartData: [
      { month: "Jan", vibration: 0.03, prob: 5, temp: 68 },
      { month: "Feb", vibration: 0.04, prob: 6, temp: 69 },
      { month: "Mar", vibration: 0.03, prob: 4, temp: 68 },
      { month: "Apr", vibration: 0.05, prob: 5, temp: 72 },
      { month: "May", vibration: 0.04, prob: 8, temp: 70 },
      { month: "Jun", vibration: 0.03, prob: 5, temp: 69 },
      { month: "Jul", vibration: 0.04, prob: 6, temp: 68 }
    ]
  },
  M103: {
    logs: [
      { date: "2026-06-01", issue: "Slew motor hydraulic leak", action: "Replaced motor pressure seals and gear oil", downtime: 3.0, prob: 0.08 }
    ],
    chartData: [
      { month: "Jan", vibration: 0.06, prob: 15, temp: 72 },
      { month: "Feb", vibration: 0.07, prob: 17, temp: 74 },
      { month: "Mar", vibration: 0.05, prob: 11, temp: 71 },
      { month: "Apr", vibration: 0.08, prob: 20, temp: 75 },
      { month: "May", vibration: 0.09, prob: 22, temp: 76 },
      { month: "Jun", vibration: 0.08, prob: 8, temp: 72 },
      { month: "Jul", vibration: 0.06, prob: 12, temp: 73 }
    ]
  }
};

export default function HistoryPage() {
  const [machineId, setMachineId] = useState("M101");
  const [data, setData] = useState(MOCK_DATA["M101"]);
  const [mounted, setMounted] = useState(false);

  // Prevent SSR hydration issues with Recharts
  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    setData(MOCK_DATA[machineId] || MOCK_DATA["M101"]);
  }, [machineId]);

  return (
    <div className="flex flex-col gap-8">
      {/* Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-brown-300/20 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-brown-900">Machine Telemetry & Maintenance History</h1>
          <p className="text-sm text-brown-700">Track logs, downtime statistics, and failure probability trends.</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs font-bold uppercase text-brown-700">Unit ID:</label>
          <select
            value={machineId}
            onChange={(e) => setMachineId(e.target.value)}
            className="px-3 py-2 bg-surface border border-brown-300 rounded-xl text-sm font-semibold text-brown-900 focus:outline-none focus:border-brown-500"
          >
            <option value="M101">M101 (Excavator)</option>
            <option value="M102">M102 (Excavator)</option>
            <option value="M103">M103 (Excavator)</option>
          </select>
        </div>
      </div>

      {/* Main Grid */}
      {/* Main Grid */}
      <AnimatePresence mode="wait">
        <motion.div
          key={machineId}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -15 }}
          transition={{ duration: 0.35, ease: "easeOut" }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-8"
        >
          {/* Left column: Historical Timeline */}
          <div className="lg:col-span-1 flex flex-col gap-6">
            <h3 className="text-xl font-bold text-brown-900 flex items-center gap-2">
              <Clock className="w-5 h-5 text-brown-700" />
              Maintenance Log Timeline
            </h3>

            <div className="flex flex-col gap-4 relative pl-4 border-l border-brown-300/30">
              {data.logs.map((log, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.08, duration: 0.35 }}
                  className="relative flex flex-col gap-2 bg-surface p-4 border border-brown-300/30 rounded-xl shadow-sm hover:shadow-md transition-shadow"
                >
                  {/* Timeline dot */}
                  <div className="absolute -left-[21px] top-5 w-2.5 h-2.5 bg-brown-700 border-2 border-background rounded-full" />
                  
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-extrabold text-brown-500">{log.date}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      log.prob > 0.5 ? "bg-danger/10 text-danger" : "bg-success/10 text-success"
                    }`}>
                      Prob: {Math.round(log.prob * 100)}%
                    </span>
                  </div>
                  
                  <h4 className="text-sm font-bold text-brown-900">{log.issue}</h4>
                  <p className="text-xs text-brown-700 leading-relaxed"><span className="font-semibold">Action:</span> {log.action}</p>
                  <div className="flex items-center justify-between text-[11px] text-brown-700 border-t border-brown-300/10 pt-2 mt-1">
                    <span className="flex items-center gap-1"><Wrench className="w-3 h-3" /> Scheduled Downtime</span>
                    <span className="font-bold text-brown-900">{log.downtime} Hours</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Right column: Charts and Telemetry trends */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            <h3 className="text-xl font-bold text-brown-900 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-brown-700" />
              Vibration & Failure Probability Trend
            </h3>

            <div className="bento-card flex flex-col gap-6 h-[400px] justify-center items-center">
              {mounted ? (
                <ResponsiveContainer width="100%" height="90%">
                  <LineChart data={data.chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F0E6DA" />
                    <XAxis dataKey="month" stroke="#3B2A1E" fontSize={12} />
                    <YAxis yAxisId="left" stroke="#3B2A1E" fontSize={12} label={{ value: 'Failure Probability (%)', angle: -90, position: 'insideLeft', offset: 10, fill: '#3B2A1E' }} />
                    <YAxis yAxisId="right" orientation="right" stroke="#A2402D" fontSize={12} label={{ value: 'Vibration (mm)', angle: 90, position: 'insideRight', offset: 10, fill: '#A2402D' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#FAF6F1', borderColor: '#C9A87C' }} />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="prob" name="Failure Probability (%)" stroke="#6B4A32" strokeWidth={2.5} activeDot={{ r: 8 }} />
                    <Line yAxisId="right" type="monotone" dataKey="vibration" name="Vibration (mm)" stroke="#A2402D" strokeWidth={2.5} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="w-8 h-8 border-2 border-brown-700 border-t-transparent rounded-full animate-spin" />
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bento-card flex flex-col gap-2">
                <h4 className="font-bold text-brown-900 text-sm">Overall Risk Assessment</h4>
                <p className="text-xs text-brown-700 leading-relaxed">
                  The machine is currently within operating margins. The main pump housing vibration has decreased to **0.05 mm** post shaft alignment in May. Continue to monitor rotational speeds over **1800 RPM** during high-torque loading.
                </p>
              </div>
              <div className="bento-card flex flex-col gap-2">
                <h4 className="font-bold text-brown-900 text-sm">Preventive Recommendations</h4>
                <p className="text-xs text-brown-700 leading-relaxed font-medium">
                  1. Conduct lubricant grease analysis every 250 operating hours.<br />
                  2. Re-verify coupling insert bolt torque status during standard weekly inspections.<br />
                  3. Align shaft laser if pump temperature hits 355 K (82°C).
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
