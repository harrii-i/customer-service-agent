import type { ApiMessage } from "@/types/api";
import MemoryBadge from "@/components/memory/MemoryBadge";
import SourceList from "@/components/sources/SourceList";

export default function MessageBubble({ message }: { message: ApiMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-slate-900 text-white rounded-br-sm"
            : "bg-white text-slate-800 border border-slate-200 rounded-bl-sm"
        }`}
      >
        {message.content}
      </div>

      {/* Provenance belongs to the answer, so it sits under the assistant's
          bubble only. Both render nothing when there is nothing to show. */}
      {!isUser && (
        <div className="max-w-[75%]">
          <MemoryBadge memories={message.memories ?? []} />
          <SourceList sources={message.sources ?? []} />
        </div>
      )}
    </div>
  );
}
