"use client";

import { useState } from "react";
import type { Memory } from "@/types/api";

/**
 * Shows what the agent recalled about this customer from earlier
 * conversations. Renders nothing when nothing was recalled — the badge is a
 * claim about what actually happened, so an empty one would be a lie.
 *
 * Collapsed by default: the count is the interesting part, the contents are
 * there for when someone doubts it.
 */
export default function MemoryBadge({ memories }: { memories: Memory[] }) {
  const [open, setOpen] = useState(false);
  if (memories.length === 0) return null;

  const label = memories.length === 1 ? "1 memory used" : `${memories.length} memories used`;

  return (
    <div className="mt-1.5">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="inline-flex items-center gap-1 rounded-full border border-violet-200 bg-violet-50 px-2.5 py-1 text-xs font-medium text-violet-800 hover:bg-violet-100"
      >
        <span aria-hidden>🧠</span>
        {label}
        <span aria-hidden className="text-violet-400">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <ul className="mt-1.5 space-y-1">
          {memories.map((memory, i) => (
            <li
              key={memory.id ?? i}
              className="rounded-lg border border-violet-100 bg-violet-50/60 px-3 py-1.5 text-xs text-violet-900"
            >
              <span className="mr-1.5 rounded bg-violet-200/70 px-1.5 py-0.5 font-medium uppercase tracking-wide text-[10px] text-violet-800">
                {memory.type}
              </span>
              {memory.content}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
