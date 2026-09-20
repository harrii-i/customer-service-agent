"use client";

import { useCallback, useEffect, useState } from "react";
import {
  createConversation,
  getConversation,
  listConversations,
  sendChatMessage,
} from "@/lib/api";
import { useAuth } from "@/components/auth/AuthProvider";
import type { ApiMessage, ConversationSummary } from "@/types/api";
import ConversationSidebar from "@/components/conversations/ConversationSidebar";
import MemoryPanel from "@/components/memory/MemoryPanel";
import ChatInput from "./ChatInput";
import MessageList from "./MessageList";

export default function ChatWindow() {
  const { user, signOut } = useAuth();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ApiMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showMemories, setShowMemories] = useState(false);

  // No identity bootstrap any more: the session already knows who this is.
  useEffect(() => {
    listConversations()
      .then(setConversations)
      .catch((e: Error) => setError(e.message));
  }, []);

  const openConversation = useCallback(async (conversationId: string) => {
    setActiveId(conversationId);
    setError(null);
    try {
      const detail = await getConversation(conversationId);
      setMessages(detail.messages);
    } catch (e) {
      setMessages([]);
      setError((e as Error).message);
    }
  }, []);

  async function handleCreate() {
    try {
      const { conversation_id } = await createConversation();
      setConversations(await listConversations());
      setMessages([]);
      setActiveId(conversation_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function handleSend(text: string) {
    if (!activeId) return;
    setError(null);

    // Optimistic echo so the input feels responsive; replaced by server state
    // once the round-trip completes.
    const optimistic: ApiMessage = {
      id: `pending-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
      memories: [],
      sources: [],
    };
    setMessages((prev) => [...prev, optimistic]);
    setPending(true);

    try {
      await sendChatMessage(activeId, text);
      const detail = await getConversation(activeId);
      setMessages(detail.messages);
      setConversations(await listConversations());
    } catch (e) {
      setMessages((prev) => prev.filter((m) => m.id !== optimistic.id));
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex h-screen flex-col bg-slate-100">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3">
        <h1 className="text-sm font-semibold text-slate-800">
          Customer Support AI
        </h1>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowMemories(true)}
            className="rounded-lg border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100"
          >
            🧠 What you remember
          </button>
          <span className="text-xs text-slate-500">{user?.name}</span>
          <button
            onClick={signOut}
            className="rounded-lg px-2.5 py-1 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-800"
          >
            Sign out
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <ConversationSidebar
          conversations={conversations}
          activeId={activeId}
          onSelect={openConversation}
          onCreate={handleCreate}
        />

        <main className="flex min-w-0 flex-1 flex-col">
          {error && (
            <p className="border-b border-red-200 bg-red-50 px-5 py-2 text-sm text-red-700">
              {error}
            </p>
          )}

          {activeId ? (
            <>
              <MessageList messages={messages} pending={pending} />
              <ChatInput disabled={pending} onSend={handleSend} />
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center text-sm text-slate-400">
              Start a new conversation to begin.
            </div>
          )}
        </main>
      </div>

      {showMemories && <MemoryPanel onClose={() => setShowMemories(false)} />}
    </div>
  );
}
