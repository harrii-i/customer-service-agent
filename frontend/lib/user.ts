const STORAGE_KEY = "csam_user_id";

/**
 * MVP identity. The browser holds a user id in localStorage and sends it with
 * each request; there is no authentication yet. Its only job is to demonstrate
 * per-user data and memory isolation.
 */
export async function getOrCreateUserId(apiBase: string): Promise<string> {
  const existing = localStorage.getItem(STORAGE_KEY);
  if (existing) return existing;

  const res = await fetch(`${apiBase}/users`, { method: "POST" });
  if (!res.ok) throw new Error("Could not create a user");

  const { user_id } = (await res.json()) as { user_id: string };
  localStorage.setItem(STORAGE_KEY, user_id);
  return user_id;
}
