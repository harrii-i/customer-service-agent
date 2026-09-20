"use client";

import Link from "next/link";
import { useState } from "react";

/** Eye / eye-with-slash, inline so the app takes on no icon dependency for
 *  two glyphs. `aria-hidden` because the button beside it is already
 *  labelled. */
function EyeIcon({ off }: { off: boolean }) {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-4 w-4"
    >
      <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
      {off && <path d="m3 3 18 18" />}
    </svg>
  );
}

export interface Field {
  name: string;
  label: string;
  type: string;
  autoComplete: string;
}

/**
 * The shared shell for sign in and sign up. One component so the two pages
 * cannot drift apart in how they report errors or handle a slow network.
 */
export default function AuthForm({
  title,
  subtitle,
  fields,
  submitLabel,
  onSubmit,
  footer,
}: {
  title: string;
  subtitle: string;
  fields: Field[];
  submitLabel: string;
  onSubmit: (values: Record<string, string>) => Promise<void>;
  footer: { text: string; linkLabel: string; href: string };
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Which password fields are currently shown. Keyed by field name rather than
  // a single flag, so a form with more than one password reveals them
  // independently. Always starts hidden: revealing is the deliberate act.
  const [revealed, setRevealed] = useState<Set<string>>(new Set());

  function toggleReveal(name: string) {
    setRevealed((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await onSubmit(values);
    } catch (e) {
      // The backend's `detail`, which is written for users and never carries
      // internal error text.
      setError((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-100 px-4">
      <div className="w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-7 shadow-sm">
        <h1 className="text-lg font-semibold text-slate-900">{title}</h1>
        <p className="mt-1 text-sm text-slate-500">{subtitle}</p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          {fields.map((field) => {
            const isPassword = field.type === "password";
            const shown = revealed.has(field.name);
            return (
              <div key={field.name}>
                <label
                  htmlFor={field.name}
                  className="block text-xs font-medium text-slate-700"
                >
                  {field.label}
                </label>
                <div className="relative mt-1">
                  <input
                    id={field.name}
                    name={field.name}
                    // Swapping the type is what reveals it — the value itself
                    // is never copied anywhere to be displayed.
                    type={isPassword && shown ? "text" : field.type}
                    autoComplete={field.autoComplete}
                    required
                    value={values[field.name] ?? ""}
                    onChange={(event) =>
                      setValues((v) => ({
                        ...v,
                        [field.name]: event.target.value,
                      }))
                    }
                    className={`w-full rounded-lg border border-slate-300 py-2 pl-3 text-sm text-slate-900 outline-none focus:border-slate-500 ${
                      isPassword ? "pr-10" : "pr-3"
                    }`}
                  />
                  {isPassword && (
                    <button
                      // Not a submit button: inside a form, a bare <button>
                      // submits it, so clicking the eye would try to sign in.
                      type="button"
                      onClick={() => toggleReveal(field.name)}
                      // Never focusable by Tab: someone tabbing from the
                      // password field expects the submit button next, not a
                      // control that exposes what they just typed.
                      tabIndex={-1}
                      aria-pressed={shown}
                      aria-label={shown ? "Hide password" : "Show password"}
                      title={shown ? "Hide password" : "Show password"}
                      className="absolute inset-y-0 right-0 flex items-center px-3 text-slate-400 hover:text-slate-700"
                    >
                      <EyeIcon off={shown} />
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          {error && (
            <p
              role="alert"
              className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700"
            >
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            {busy ? "Please wait..." : submitLabel}
          </button>
        </form>

        <p className="mt-5 text-center text-xs text-slate-500">
          {footer.text}{" "}
          <Link href={footer.href} className="font-medium text-slate-900 underline">
            {footer.linkLabel}
          </Link>
        </p>
      </div>
    </main>
  );
}
