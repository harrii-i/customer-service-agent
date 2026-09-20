export type Role = "user" | "assistant";

/** The signed-in account. Never carries a password or its hash — the backend
 *  has no field to send one. */
export interface User {
  id: string;
  name: string;
  email: string;
}

/** What the agent actually recalled and cited for a message. The UI must
 *  never invent either, and must never show one the agent did not receive. */
export interface Memory {
  id?: string | null;
  content: string;
  type: string;
}

export interface Source {
  title: string;
  source: string;
  snippet?: string | null;
}

export interface ApiMessage {
  id: string;
  role: Role;
  content: string;
  created_at: string;
  /** Persisted alongside the message, so provenance survives a reload.
   *  Always empty for user messages. */
  memories: Memory[];
  sources: Source[];
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ApiMessage[];
}

export interface ChatResponse {
  conversation_id: string;
  message: string;
  memory_used: boolean;
  memories: Memory[];
  sources: Source[];
}
