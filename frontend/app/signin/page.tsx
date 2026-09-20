"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import AuthForm from "@/components/auth/AuthForm";
import { useAuth } from "@/components/auth/AuthProvider";

export default function SignInPage() {
  const { signIn, user, loading } = useAuth();
  const router = useRouter();

  // Already signed in — no reason to show a sign-in form.
  useEffect(() => {
    if (!loading && user) router.replace("/chat");
  }, [loading, user, router]);

  return (
    <AuthForm
      title="Sign in"
      subtitle="Welcome back to Customer Support AI."
      fields={[
        { name: "email", label: "Email", type: "email", autoComplete: "email" },
        {
          name: "password",
          label: "Password",
          type: "password",
          autoComplete: "current-password",
        },
      ]}
      submitLabel="Sign in"
      onSubmit={async (values) => {
        await signIn(values.email, values.password);
        router.replace("/chat");
      }}
      footer={{
        text: "No account yet?",
        linkLabel: "Create one",
        href: "/signup",
      }}
    />
  );
}
