"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

export interface AuthUser {
  uid: string;
  email: string;
  displayName: string;
  photoURL?: string;
  role: "user" | "admin";
}

interface AuthContextType {
  user: AuthUser | null;
  role: "user" | "admin";
  token: string | null;
  loading: boolean;
  loginWithEmail: (emailOrUser: string, password: string) => Promise<void>;
  loginWithGoogle: () => Promise<void>;
  registerWithEmail: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [role, setRole] = useState<"user" | "admin">("user");
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Retrieve user session on mount (auto-login default Admin Luffy if not set)
  useEffect(() => {
    setLoading(true);
    let savedToken = localStorage.getItem("fm_jwt_token");
    let savedRole = localStorage.getItem("fm_user_role") as "user" | "admin";
    let savedName = localStorage.getItem("fm_user_name");

    if (!savedToken || !savedRole || !savedName) {
      savedToken = "mock-firebase-jwt-token-luffy";
      savedRole = "admin";
      savedName = "Luffy";
      localStorage.setItem("fm_jwt_token", savedToken);
      localStorage.setItem("fm_user_role", savedRole);
      localStorage.setItem("fm_user_name", savedName);
    }

    setRole(savedRole);
    setToken(savedToken);
    setUser({
      uid: `user-${savedName.toLowerCase()}`,
      email: `${savedName.toLowerCase()}@factorymind.ai`,
      displayName: savedName,
      role: savedRole
    });
    setLoading(false);
  }, []);

  const loginWithEmail = async (emailOrUser: string, password: string) => {
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: emailOrUser, password })
      });

      if (res.ok) {
        const data = await res.json();
        localStorage.setItem("fm_jwt_token", data.token);
        localStorage.setItem("fm_user_role", data.role);
        localStorage.setItem("fm_user_name", data.username);

        setToken(data.token);
        setRole(data.role);
        
        setUser({
          uid: `user-${data.username.toLowerCase()}`,
          email: `${data.username.toLowerCase()}@factorymind.ai`,
          displayName: data.username,
          role: data.role
        });
      } else {
        const err = await res.json();
        throw new Error(err.detail || "Invalid login credentials.");
      }
    } catch (e: any) {
      setLoading(false);
      throw e;
    }
    setLoading(false);
  };

  const loginWithGoogle = async () => {
    // Standard mock option for Google sign-in since we removed external identity providers
    setLoading(true);
    try {
      await loginWithEmail("onepiece", "luffy");
    } catch (e: any) {
      setLoading(false);
      throw e;
    }
    setLoading(false);
  };

  const registerWithEmail = async (email: string, password: string, name: string) => {
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          username: email.split("@")[0], 
          password, 
          display_name: name,
          role: "user" 
        })
      });

      if (res.ok) {
        const data = await res.json();
        localStorage.setItem("fm_jwt_token", data.token);
        localStorage.setItem("fm_user_role", data.role);
        localStorage.setItem("fm_user_name", data.username);

        setToken(data.token);
        setRole(data.role);
        
        setUser({
          uid: `user-${data.username.toLowerCase()}`,
          email: `${data.username.toLowerCase()}@factorymind.ai`,
          displayName: data.username,
          role: data.role
        });
      } else {
        const err = await res.json();
        throw new Error(err.detail || "Registration failed.");
      }
    } catch (e: any) {
      setLoading(false);
      throw e;
    }
    setLoading(false);
  };

  const logout = async () => {
    setLoading(true);
    localStorage.removeItem("fm_jwt_token");
    localStorage.removeItem("fm_user_role");
    localStorage.removeItem("fm_user_name");
    setUser(null);
    setRole("user");
    setToken(null);
    setLoading(false);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        role,
        token,
        loading,
        loginWithEmail,
        loginWithGoogle,
        registerWithEmail,
        logout
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
