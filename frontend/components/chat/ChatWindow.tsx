"use client";

import { useCallback, useEffect, useState } from "react";
import {
  API_BASE,
  createConversation,
  getConversation,
  listConversations,
  sendChatMessage,
} from "@/lib/api";
import { getOrCreateUserId } from "@/lib/user";
import type { ApiMessage, ConversationSummary } from "@/types/api";
import ConversationSidebar from "@/components/conversations/ConversationSidebar";
import ChatInput from "./ChatInput";
import MessageList from "./MessageList";

export default function ChatWindow() {
  const [userId, setUserId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ApiMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Establish the MVP identity, then load this user's conversations.
  useEffect(() => {
    getOrCreateUserId(API_BASE)
      .then(async (id) => {
        setUserId(id);
        setConversations(await listConversations(id));
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const openConversation = useCallback(
    async (conversationId: string, uid: string) => {
      setActiveId(conversationId);
      setError(null);
      try {
        const detail = await getConversation(conversationId, uid);
        setMessages(detail.messages);
      } catch (e) {
        setMessages([]);
        setError((e as Error).message);
      }
    },
    [],
  );

  async function handleCreate() {
    if (!userId) return;
    try {
      const { conversation_id } = await createConversation(userId);
      setConversations(await listConversations(userId));
      setMessages([]);
      setActiveId(conversation_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function handleSend(text: string) {
    if (!userId || !activeId) return;
    setError(null);

    // Optimistic echo so the input feels responsive; replaced by server state
    // once the round-trip completes.
    const optimistic: ApiMessage = {
      id: `pending-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
      // A user message never carries provenance; the server state that
      // replaces this in a moment is what carries the assistant's.
      memories: [],
      sources: [],
    };
    setMessages((prev) => [...prev, optimistic]);
    setPending(true);

    try {
      await sendChatMessage(userId, activeId, text);
      const detail = await getConversation(activeId, userId);
      setMessages(detail.messages);
      setConversations(await listConversations(userId));
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
        <span className="font-mono text-xs text-slate-400">
          {userId ? `user ${userId.slice(0, 8)}` : "connecting..."}
        </span>
      </header>

      <div className="flex min-h-0 flex-1">
        <ConversationSidebar
          conversations={conversations}
          activeId={activeId}
          onSelect={(id) => userId && openConversation(id, userId)}
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
    </div>
  );
}
