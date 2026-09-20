"use client";

import { useEffect, useState } from "react";
import { deleteMemory, forgetAllMemories, listMemories } from "@/lib/api";
import type { Memory } from "@/types/api";

/**
 * Everything the agent has stored about the signed-in customer, and the means
 * to delete any of it.
 *
 * A system that quietly remembers people should let them see what it has and
 * remove it. The badges show what was used for one reply; this shows the whole
 * store.
 */
export default function MemoryPanel({ onClose }: { onClose: () => void }) {
  const [memories, setMemories] = useState<Memory[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listMemories()
      .then(setMemories)
      .catch((e: Error) => setError(e.message));
  }, []);

  async function remove(id: string) {
    try {
      await deleteMemory(id);
      setMemories((current) => (current ?? []).filter((m) => m.id !== id));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function forgetAll() {
    try {
      await forgetAllMemories();
      setMemories([]);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div
      className="fixed inset-0 z-20 flex justify-end bg-slate-900/30"
      onClick={onClose}
    >
      <aside
        onClick={(event) => event.stopPropagation()}
        className="flex h-full w-full max-w-md flex-col border-l border-slate-200 bg-white"
      >
        <header className="flex items-center justify-between border-b border-slate-200 px-5 py-3">
          <h2 className="text-sm font-semibold text-slate-800">
            What the agent remembers
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-xs text-slate-500 hover:bg-slate-100"
          >
            Close
          </button>
        </header>

        <div className="flex-1 space-y-2 overflow-y-auto p-5">
          {error && (
            <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {error}
            </p>
          )}
          {memories === null && !error && (
            <p className="text-xs text-slate-400">Loading...</p>
          )}
          {memories?.length === 0 && (
            <p className="text-xs text-slate-400">
              Nothing remembered yet. Tell the agent about yourself and it will
              recall it in later conversations.
            </p>
          )}
          {memories?.map((memory) => (
            <div
              key={memory.id}
              className="flex items-start justify-between gap-3 rounded-lg border border-violet-100 bg-violet-50/60 px-3 py-2"
            >
              <p className="text-xs text-violet-900">
                <span className="mr-1.5 rounded bg-violet-200/70 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-violet-800">
                  {memory.type}
                </span>
                {memory.content}
              </p>
              {memory.id && (
                <button
                  onClick={() => remove(memory.id!)}
                  aria-label="Forget this"
                  className="shrink-0 rounded px-1.5 py-0.5 text-xs text-violet-500 hover:bg-violet-200/60 hover:text-violet-900"
                >
                  Forget
                </button>
              )}
            </div>
          ))}
        </div>

        {!!memories?.length && (
          <footer className="border-t border-slate-200 p-4">
            <button
              onClick={forgetAll}
              className="w-full rounded-lg border border-red-200 px-3 py-2 text-xs font-medium text-red-700 hover:bg-red-50"
            >
              Forget everything about me
            </button>
          </footer>
        )}
      </aside>
    </div>
  );
}
