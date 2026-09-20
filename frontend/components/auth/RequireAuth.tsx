"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useAuth } from "./AuthProvider";

/**
 * Gate for pages that need a signed-in account.
 *
 * This is a usability guard, not a security boundary. It stops an
 * unauthenticated person seeing the chat shell; it is the backend's 401 that
 * actually protects the data. Anyone can edit client-side state, so nothing
 * here is trusted.
 */
export default function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) router.replace("/signin");
  }, [loading, user, router]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-slate-400">
        Loading...
      </div>
    );
  }
  if (!user) return null; // redirecting
  return <>{children}</>;
}
