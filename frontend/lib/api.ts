import type {
  ChatResponse,
  ConversationDetail,
  ConversationSummary,
  Memory,
  User,
} from "@/types/api";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Thrown for any non-2xx response, carrying the backend's own message where
 *  it has one. The backend never puts internal errors in `detail`, so this is
 *  safe to show a user. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** Every call sends the auth cookie. It is httpOnly, so this code cannot read
 *  the token — which is the point: an injected script cannot steal it either.
 *  The Gemini key and every LLM call stay behind this boundary, in FastAPI. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError("Could not reach the server. Is the backend running?", 0);
  }

  if (!res.ok) {
    const detail = await res
      .json()
      .then((body) => {
        const value = body?.detail;
        // FastAPI validation errors are a list of objects, not a string.
        if (typeof value === "string") return value;
        if (Array.isArray(value) && value[0]?.msg) return value[0].msg as string;
        return undefined;
      })
      .catch(() => undefined);
    throw new ApiError(detail ?? `Request failed (${res.status})`, res.status);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// --- auth ------------------------------------------------------------------

export function signUp(name: string, email: string, password: string) {
  return request<User>("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ name, email, password }),
  });
}

export function signIn(email: string, password: string) {
  return request<User>("/auth/signin", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function signOut() {
  return request<void>("/auth/logout", { method: "POST" });
}

export function fetchMe() {
  return request<User>("/auth/me");
}

// --- conversations ---------------------------------------------------------
// No user id anywhere: the server takes it from the token.

export function listConversations() {
  return request<ConversationSummary[]>("/conversations");
}

export function createConversation() {
  return request<{ conversation_id: string }>("/conversations", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function getConversation(conversationId: string) {
  return request<ConversationDetail>(`/conversations/${conversationId}`);
}

export function sendChatMessage(conversationId: string, message: string) {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({ conversation_id: conversationId, message }),
  });
}

// --- memories --------------------------------------------------------------

export function listMemories() {
  return request<Memory[]>("/memories");
}

export function deleteMemory(memoryId: string) {
  return request<void>(`/memories/${memoryId}`, { method: "DELETE" });
}

export function forgetAllMemories() {
  return request<void>("/memories", { method: "DELETE" });
}
