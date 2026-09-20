"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import * as api from "@/lib/api";
import type { User } from "@/types/api";

interface AuthState {
  user: User | null;
  /** True until the session has been checked. Pages must wait for this before
   *  deciding anyone is unauthenticated, or a refresh flashes the sign-in page
   *  at someone who is perfectly well signed in. */
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Session restore. The token is an httpOnly cookie, so this code cannot
  // read it — it asks the server who the cookie belongs to instead. That is
  // what makes a refresh keep you signed in without exposing the token to
  // JavaScript.
  useEffect(() => {
    api
      .fetchMe()
      .then(setUser)
      .catch(() => setUser(null)) // no session, or it expired
      .finally(() => setLoading(false));
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    setUser(await api.signIn(email, password));
  }, []);

  const signUp = useCallback(
    async (name: string, email: string, password: string) => {
      setUser(await api.signUp(name, email, password));
    },
    [],
  );

  const signOut = useCallback(async () => {
    try {
      await api.signOut();
    } finally {
      // Clear locally even if the request failed: the user asked to be signed
      // out, and leaving them looking signed in is the worse outcome.
      setUser(null);
    }
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, signIn, signUp, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}
