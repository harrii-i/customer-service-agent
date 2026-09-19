import type { ConversationSummary } from "@/types/api";

export default function ConversationItem({
  conversation,
  active,
  onSelect,
}: {
  conversation: ConversationSummary;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <button
      onClick={() => onSelect(conversation.id)}
      className={`w-full truncate rounded-lg px-3 py-2 text-left text-sm ${
        active
          ? "bg-slate-200 font-medium text-slate-900"
          : "text-slate-600 hover:bg-slate-100"
      }`}
    >
      {conversation.title ?? "New conversation"}
    </button>
  );
}
