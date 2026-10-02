import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createJiti } from "jiti";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function sizeLimitToBytes(value) {
  if (typeof value === "number") return value;
  const match = /^(\d+(?:\.\d+)?)\s*(b|kb|mb|gb)$/i.exec(value.trim());
  assert.ok(match, `unparseable SizeLimit: ${value}`);
  const units = { b: 1, kb: 1024, mb: 1024 ** 2, gb: 1024 ** 3 };
  return Number(match[1]) * units[match[2].toLowerCase()];
}

test("scopes Next.js output file tracing to the pi-web package", async () => {
  const config = await createJiti(import.meta.url).import("../next.config.ts", { default: true });

  assert.equal(config.outputFileTracingRoot, projectRoot);
});

test("raises the proxy body buffer above the upload route's 100 MB request cap", async () => {
  const config = await createJiti(import.meta.url).import("../next.config.ts", { default: true });

  assert.ok(sizeLimitToBytes(config.experimental.proxyClientMaxBodySize) > 100 * 1024 * 1024);
});

test("development chunks prohibit storage without changing production asset caching", async () => {
  const config = await createJiti(import.meta.url).import("../next.config.ts", { default: true });
  const original = process.env.NODE_ENV;
  try {
    process.env.NODE_ENV = "development";
    const development = await config.headers();
    const chunks = development.find((rule) => rule.source === "/_next/static/:path*");
    assert.ok(chunks);
    const headers = Object.fromEntries(chunks.headers.map(({ key, value }) => [key.toLowerCase(), value]));
    assert.match(headers["cache-control"], /(?:^|,\s*)no-store(?:,|$)/);
    assert.match(headers["cache-control"], /(?:^|,\s*)private(?:,|$)/);
    assert.equal(headers["cdn-cache-control"], "no-store");
    assert.equal(headers["cloudflare-cdn-cache-control"], "no-store");

    process.env.NODE_ENV = "production";
    const production = await config.headers();
    assert.equal(production.some((rule) => rule.source === "/_next/static/:path*"), false);
    assert.deepEqual(development.filter((rule) => rule !== chunks), production);
  } finally {
    if (original === undefined) delete process.env.NODE_ENV;
    else process.env.NODE_ENV = original;
  }
});
