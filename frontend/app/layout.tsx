"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Wrench, LogOut, User, Menu, Play } from "lucide-react";
import Lenis from "lenis";
import { AuthProvider, useAuth } from "./context/AuthContext";
import "./globals.css";

function LayoutContent({ children }: { children: React.ReactNode }) {
  const { user, role, logout, loading } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  // Initialize Lenis smooth scroll
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    });

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);

    return () => lenis.destroy();
  }, []);

  // Auto-route to Ask Assistant if landing on root
  useEffect(() => {
    if (!loading && pathname === "/") {
      router.push("/ask");
    }
  }, [loading, pathname, router]);

  const toggleRole = () => {
    const nextRole = role === "admin" ? "user" : "admin";
    const nextName = nextRole === "admin" ? "Luffy" : "Zoro";
    localStorage.setItem("fm_jwt_token", `mock-firebase-jwt-token-${nextRole}`);
    localStorage.setItem("fm_user_role", nextRole);
    localStorage.setItem("fm_user_name", nextName);
    window.location.reload();
  };

  const navItems = [
    { name: "Ask Assistant", path: "/ask" },
    { name: "Machine History", path: "/history" },
    { name: "Reports", path: "/reports" },
    ...(role === "admin" ? [
      { name: "Admin Dashboard", path: "/admin" },
      { name: "RAG Evaluation", path: "/eval" }
    ] : []),
  ];

  return (
    <div className="flex flex-col min-h-screen">
      {/* Navigation Bar */}
      <header className="sticky top-0 z-50 bg-[#FAF6F1]/80 backdrop-blur-md border-b border-brown-300/30">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href={user ? "/ask" : "/login"} className="flex items-center gap-2.5">
            <div className="bg-brown-700 text-background p-2 rounded-lg">
              <Wrench className="w-5 h-5" />
            </div>
            <div>
              <span className="font-extrabold text-xl tracking-tight text-brown-900 block leading-none">
                FactoryMind <span className="text-brown-500 font-normal">AI</span>
              </span>
              <span className="text-[10px] text-brown-750 tracking-wider font-semibold uppercase mt-1 block">
                Intelligent Maintenance
              </span>
            </div>
          </Link>

          {user && (
            <>
              {/* Nav links */}
              <nav className="hidden md:flex items-center gap-1">
                {navItems.map((item) => {
                  const isActive = pathname === item.path;
                  return (
                    <Link
                      key={item.path}
                      href={item.path}
                      className={`px-4 py-2 rounded-xl text-xs font-bold transition-all duration-200 ${
                        isActive
                          ? "bg-brown-900 text-white shadow-sm"
                          : "text-brown-750 hover:bg-surface-alt/50"
                      }`}
                    >
                      {item.name}
                    </Link>
                  );
                })}
              </nav>

              {/* Profile card badge & Role Switcher */}
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-brown-300/25 text-xs font-bold text-brown-800 shadow-sm">
                  <User className="w-3.5 h-3.5 text-brown-700 shrink-0" />
                  <span>{user.displayName}</span>
                  <span className="px-1.5 py-0.5 rounded text-[8px] font-extrabold uppercase bg-brown-700/10 text-brown-700">
                    {user.role}
                  </span>
                </div>
                
                <button
                  onClick={toggleRole}
                  className="px-3 py-1.5 rounded-xl border border-brown-300/35 text-xs font-bold text-brown-750 hover:bg-brown-900 hover:text-white transition-colors"
                >
                  Switch to {role === "admin" ? "Engineer" : "Admin"}
                </button>
              </div>
            </>
          )}

          {!user && (
            <div className="flex items-center gap-3">
              <span className="text-xs font-bold text-brown-700 uppercase tracking-widest animate-pulse">
                Authentication Required
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl w-full mx-auto p-6 md:p-8 flex flex-col justify-center">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-brown-300/20 py-8 bg-surface-alt/25 mt-auto">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-brown-750">
          <div>
            <p className="font-bold text-brown-900">FactoryMind AI</p>
            <p className="mt-1">From Reactive Repairs to Grounded Industrial RAG.</p>
          </div>
          <div>
            <p>© {new Date().getFullYear()} FactoryMind. Hyundai R215L Excavator Series.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          <LayoutContent>{children}</LayoutContent>
        </AuthProvider>
      </body>
    </html>
  );
}
