"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import AuthForm from "@/components/auth/AuthForm";
import { useAuth } from "@/components/auth/AuthProvider";

export default function SignUpPage() {
  const { signUp, user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && user) router.replace("/chat");
  }, [loading, user, router]);

  return (
    <AuthForm
      title="Create an account"
      subtitle="The agent remembers you across conversations."
      fields={[
        { name: "name", label: "Name", type: "text", autoComplete: "name" },
        { name: "email", label: "Email", type: "email", autoComplete: "email" },
        {
          name: "password",
          label: "Password",
          type: "password",
          autoComplete: "new-password",
        },
      ]}
      submitLabel="Sign up"
      onSubmit={async (values) => {
        await signUp(values.name, values.email, values.password);
        router.replace("/chat");
      }}
      footer={{
        text: "Already have an account?",
        linkLabel: "Sign in",
        href: "/signin",
      }}
    />
  );
}
