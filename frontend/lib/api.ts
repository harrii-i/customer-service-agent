import type {
  ChatResponse,
  ConversationDetail,
  ConversationSummary,
} from "@/types/api";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** The Gemini key and every LLM call live behind this boundary, in FastAPI.
 *  The browser only ever talks to our own backend. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!res.ok) {
    const detail = await res
      .json()
      .then((body) => body?.detail as string | undefined)
      .catch(() => undefined);
    throw new Error(detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export function listConversations(userId: string) {
  return request<ConversationSummary[]>(`/users/${userId}/conversations`);
}

export function createConversation(userId: string) {
  return request<{ conversation_id: string }>("/conversations", {
    method: "POST",
    body: JSON.stringify({ user_id: userId }),
  });
}

export function getConversation(conversationId: string, userId: string) {
  return request<ConversationDetail>(
    `/conversations/${conversationId}?user_id=${userId}`,
  );
}

export function sendChatMessage(
  userId: string,
  conversationId: string,
  message: string,
) {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      conversation_id: conversationId,
      message,
    }),
  });
}
