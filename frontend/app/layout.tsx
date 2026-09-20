import type { Metadata } from "next";
import { AuthProvider } from "@/components/auth/AuthProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "CSAM — Customer Support AI",
  description: "Customer support agent with long-term memory",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">
        {/* One session for the whole app, restored once on load rather than
            re-checked by every page. */}
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
