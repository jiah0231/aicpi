import assert from "node:assert/strict";
import { resolve } from "node:path";
import test from "node:test";
import { createJiti } from "jiti";

const { readArchiveEntry } = await createJiti(import.meta.url).import("./grounding-safety-extension.ts");

test("ZIP-capable tar reads an image without invoking a second extractor", async () => {
  const image = Buffer.from("image bytes");
  const calls = [];
  const actual = await readArchiveEntry("images.zip", "visible/one.png", async (...args) => {
    calls.push(args);
    return image;
  });
  assert.equal(actual, image);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], "tar");
  assert.deepEqual(calls[0][1], ["-xOf", resolve("images.zip"), "--", "visible/one.png"]);
});

test("GNU tar ZIP failure falls back to bounded, literal unzip extraction", async () => {
  const image = Buffer.from("image bytes");
  const calls = [];
  const actual = await readArchiveEntry("images with spaces.zip", "visible/one[1]*?.png", async (...args) => {
    calls.push(args);
    if (args[1][0] === "--version") return Buffer.from("tar (GNU tar) 1.35");
    if (args[0] === "tar") throw new Error("This does not look like a tar archive");
    return image;
  });
  assert.equal(actual, image);
  assert.deepEqual(calls.map(([command]) => command), ["tar", "tar", "unzip"]);
  assert.deepEqual(calls[0][1], ["--version"]);
  assert.deepEqual(calls[1][1], ["-xOf", resolve("images with spaces.zip"), "--no-wildcards", "--", "visible/one[1]*?.png"]);
  assert.deepEqual(calls[2][1], ["-p", resolve("images with spaces.zip"), "visible/one\\[1\\]\\*\\?.png"]);
  assert.equal(calls[0][2].maxBuffer, 64 * 1024);
  assert.equal(calls[0][2].timeout, 5_000);
  for (const [, , options] of calls.slice(1)) {
    assert.deepEqual(options, {
      encoding: "buffer", maxBuffer: 64 * 1024 * 1024,
      timeout: 30_000, killSignal: "SIGKILL", windowsHide: true,
    });
  }
});

test("BSD tar receives an escaped literal member pattern", async () => {
  const image = Buffer.from("image bytes");
  const calls = [];
  const actual = await readArchiveEntry("images.zip", "visible/one[1]*?.png", async (...args) => {
    calls.push(args);
    return args[1][0] === "--version" ? Buffer.from("bsdtar 3.7.2 - libarchive 3.7.2") : image;
  });
  assert.equal(actual, image);
  assert.deepEqual(calls.map(([command]) => command), ["tar", "tar"]);
  assert.deepEqual(calls[1][1], ["-xOf", resolve("images.zip"), "--", "visible/one\\[1\\]\\*\\?.png"]);
});

test("unknown tar implementations cannot expand wildcard members before ZIP fallback", async () => {
  const calls = [];
  const image = Buffer.from("image bytes");
  const actual = await readArchiveEntry("images.ZIP", "visible/one?.png", async (...args) => {
    calls.push(args);
    return args[0] === "tar" ? Buffer.from("unrecognized tar") : image;
  });
  assert.equal(actual, image);
  assert.deepEqual(calls.map(([command, args]) => [command, args]), [
    ["tar", ["--version"]],
    ["unzip", ["-p", resolve("images.ZIP"), "visible/one\\?.png"]],
  ]);
});

test("non-ZIP failures do not invoke unzip", async () => {
  const commands = [];
  await assert.rejects(readArchiveEntry("images.tar", "visible/missing.png", async (command) => {
    commands.push(command);
    throw new Error("member not found");
  }), /Could not read visible\/missing\.png from images\.tar: tar: member not found/);
  assert.deepEqual(commands, ["tar"]);
});

test("archive failure reports both extractors and never returns partial bytes", async () => {
  await assert.rejects(
    readArchiveEntry("images.zip", "visible/missing.png", async (command) => {
      throw new Error(command === "tar" ? "unsupported archive format" : "filename not matched");
    }),
    /Could not read visible\/missing\.png from images\.zip: tar: unsupported archive format; unzip: filename not matched/,
  );
});
