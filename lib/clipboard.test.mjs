import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, {
  interopDefault: true,
  moduleCache: false,
});
const { copyText } = await jiti.import("./clipboard.ts");

function installClipboardDom({ writeText, execCommand }) {
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const previousDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  const children = [];
  const body = {
    appendChild(node) {
      node.parentNode = body;
      children.push(node);
    },
    removeChild(node) {
      const index = children.indexOf(node);
      if (index >= 0) children.splice(index, 1);
      node.parentNode = null;
    },
  };
  const document = {
    body,
    createElement() {
      return {
        value: "",
        style: {},
        parentNode: null,
        setAttribute() {},
        focus() {},
        select() {},
      };
    },
    execCommand,
  };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { clipboard: { writeText } },
  });
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: document,
  });
  return {
    children,
    restore() {
      if (previousNavigator) Object.defineProperty(globalThis, "navigator", previousNavigator);
      else delete globalThis.navigator;
      if (previousDocument) Object.defineProperty(globalThis, "document", previousDocument);
      else delete globalThis.document;
    },
  };
}

test("falls back to legacy copy after Clipboard API permission rejection", async () => {
  let apiCalls = 0;
  let legacyCalls = 0;
  const dom = installClipboardDom({
    writeText: async () => {
      apiCalls += 1;
      throw new Error("Write permission denied");
    },
    execCommand: (command) => {
      legacyCalls += 1;
      assert.equal(command, "copy");
      return true;
    },
  });
  try {
    await copyText("hello");
    assert.equal(apiCalls, 1);
    assert.equal(legacyCalls, 1);
    assert.equal(dom.children.length, 0);
  } finally {
    dom.restore();
  }
});

test("cleans the fallback textarea when the legacy copy also fails", async () => {
  const dom = installClipboardDom({
    writeText: undefined,
    execCommand: () => false,
  });
  try {
    await assert.rejects(copyText("hello"), /Legacy clipboard copy was rejected/);
    assert.equal(dom.children.length, 0);
  } finally {
    dom.restore();
  }
});
