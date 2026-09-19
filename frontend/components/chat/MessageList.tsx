"use client";

import { useEffect, useRef } from "react";
import type { ApiMessage } from "@/types/api";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";

export default function MessageList({
  messages,
  pending,
}: {
  messages: ApiMessage[];
  pending: boolean;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  if (messages.length === 0 && !pending) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center text-sm text-slate-400">
        Send a message to start this conversation.
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-3 overflow-y-auto p-6">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      {pending && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}
