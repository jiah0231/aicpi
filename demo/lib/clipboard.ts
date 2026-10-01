function copyWithLegacyTextarea(text: string): void {
  if (typeof document === "undefined" || !document.body || typeof document.execCommand !== "function") {
    throw new Error("Clipboard is unavailable in this browser");
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "0";
  textarea.style.left = "-9999px";
  textarea.style.opacity = "0";

  document.body.appendChild(textarea);
  try {
    textarea.focus();
    textarea.select();
    if (!document.execCommand("copy")) {
      throw new Error("Legacy clipboard copy was rejected");
    }
  } finally {
    textarea.parentNode?.removeChild(textarea);
  }
}

export function copyText(text: string): Promise<void> {
  const clipboard = typeof navigator !== "undefined" ? navigator.clipboard : undefined;
  if (clipboard?.writeText) {
    return Promise.resolve()
      .then(() => clipboard.writeText(text))
      .catch(() => {
        copyWithLegacyTextarea(text);
      });
  }

  return Promise.resolve().then(() => copyWithLegacyTextarea(text));
}
