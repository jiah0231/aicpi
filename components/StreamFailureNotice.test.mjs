import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, { jsx: { runtime: "automatic" }, tsconfigPaths: true });
const React = await jiti.import("react");
const { renderToStaticMarkup } = await jiti.import("react-dom/server");
const { I18nProvider } = await jiti.import("@/hooks/useI18n");
const { StreamFailureNotice } = await jiti.import("./StreamFailureNotice.tsx");

function render(message) {
  return renderToStaticMarkup(React.createElement(I18nProvider, null,
    React.createElement(StreamFailureNotice, { message })));
}

test("explains incomplete replies without claiming the current run is still failing", () => {
  const html = render({ role: "assistant", stopReason: "error",
    errorMessage: "Stream ended without finish_reason", content: [{ type: "toolCall", input: { private: "hidden-input" } }] });
  assert.match(html, /later retry may already have recovered/);
  assert.match(html, /missing_finish_reason/);
  assert.match(html, /finish_marker_not_observed/);
  assert.match(html, /Tool calls in this failed reply are not executed/);
  assert.doesNotMatch(html, /hidden-input/);
});

test("does not misclassify output limits or successful recovery messages", () => {
  assert.equal(render({ role: "assistant", stopReason: "length", content: [] }), "");
  assert.equal(render({ role: "assistant", stopReason: "stop", content: [] }), "");
  assert.equal(render({ role: "assistant", stopReason: "error", errorMessage: "HTTP 503", content: [] }), "");
});
