"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  useEffect(() => {
    router.push("/ask");
  }, [router]);

  return (
    <div className="min-h-screen bg-[#FAF6F1] flex items-center justify-center">
      <div className="w-6 h-6 border-2 border-brown-700 border-t-transparent rounded-full animate-spin" />
    </div>
  );
}
