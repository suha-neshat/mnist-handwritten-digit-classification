// web/src/api.js
const BASE = "http://127.0.0.1:8000"; // hardcode for local dev

export async function predict(dataUrl) {
  // helpful debug
  console.log("POST", `${BASE}/predict`);
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl })
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status} ${res.statusText} — ${text}`);
  }
  return res.json();
}
