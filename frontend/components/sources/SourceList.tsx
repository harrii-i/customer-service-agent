"use client";

import { useState } from "react";
import type { Source } from "@/types/api";

/**
 * Shows which support documents the answer drew on. Renders nothing when the
 * agent cited nothing, so an uncited answer is visibly uncited rather than
 * quietly appearing sourced.
 */
export default function SourceList({ sources }: { sources: Source[] }) {
  const [openTitle, setOpenTitle] = useState<string | null>(null);
  if (sources.length === 0) return null;

  // One document can match on several chunks; the customer wants the document.
  const unique = sources.filter(
    (s, i) => sources.findIndex((o) => o.title === s.title) === i,
  );

  return (
    <div className="mt-1.5 flex flex-wrap gap-1.5">
      {unique.map((source) => {
        const open = openTitle === source.title;
        return (
          <div key={source.title} className="max-w-full">
            <button
              onClick={() => setOpenTitle(open ? null : source.title)}
              aria-expanded={open}
              className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-xs font-medium text-sky-800 hover:bg-sky-100"
            >
              <span aria-hidden>📚</span>
              Source: {source.title}
            </button>
            {open && source.snippet && (
              <p className="mt-1.5 rounded-lg border border-sky-100 bg-sky-50/60 px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap text-sky-900">
                {source.snippet}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
