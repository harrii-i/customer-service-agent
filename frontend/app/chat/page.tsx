import ChatWindow from "@/components/chat/ChatWindow";
import RequireAuth from "@/components/auth/RequireAuth";

export default function ChatPage() {
  return (
    <RequireAuth>
      <ChatWindow />
    </RequireAuth>
  );
}
