"use client";

import type { ConversationSummary } from "@/types/api";
import ConversationItem from "./ConversationItem";

/** Groups by day so the sidebar reads as Today / Yesterday / Earlier. */
function bucket(iso: string): string {
  const day = new Date(iso);
  const today = new Date();
  const startOf = (d: Date) =>
    new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const diffDays = Math.round((startOf(today) - startOf(day)) / 86_400_000);

  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return "Earlier";
}

export default function ConversationSidebar({
  conversations,
  activeId,
  onSelect,
  onCreate,
}: {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
}) {
  const groups = new Map<string, ConversationSummary[]>();
  for (const conversation of conversations) {
    const key = bucket(conversation.updated_at);
    groups.set(key, [...(groups.get(key) ?? []), conversation]);
  }

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-slate-50">
      <div className="p-3">
        <button
          onClick={onCreate}
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
        >
          + New Conversation
        </button>
      </div>

      <nav className="flex-1 space-y-4 overflow-y-auto px-3 pb-4">
        {conversations.length === 0 && (
          <p className="px-1 text-xs text-slate-400">No conversations yet.</p>
        )}
        {[...groups.entries()].map(([label, items]) => (
          <div key={label} className="space-y-1">
            <p className="px-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
              {label}
            </p>
            {items.map((conversation) => (
              <ConversationItem
                key={conversation.id}
                conversation={conversation}
                active={conversation.id === activeId}
                onSelect={onSelect}
              />
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
