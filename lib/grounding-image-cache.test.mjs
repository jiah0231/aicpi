import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createJiti } from "jiti";
import JSZip from "jszip";
import sharp from "sharp";

const { createGroundingSafetyExtension } = await createJiti(import.meta.url).import("./grounding-safety-extension.ts");

for (const archived of [false, true]) {
  test(`image cache preserves case-sensitive ${archived ? "ZIP member" : "Linux file"} identity`, {
    skip: !archived && process.platform !== "linux",
  }, async (t) => {
    const root = await mkdtemp(join(tmpdir(), "pi-grounding-image-cache-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    const queryPath = join(sourceDir, "queries.json");
    const archivePath = join(root, "images.zip");
    await mkdir(sourceDir);
    const archive = new JSZip();
    for (const [name, background] of [
      ["A.png", { r: 255, g: 0, b: 0 }],
      ["a.png", { r: 0, g: 0, b: 255 }],
    ]) {
      const image = await sharp({ create: { width: 64, height: 48, channels: 3, background } }).png().toBuffer();
      if (archived) archive.file(`visible/${name}`, image);
      else await writeFile(join(sourceDir, name), image);
    }
    if (archived) await writeFile(archivePath, await archive.generateAsync({ type: "nodebuffer" }));
    await writeFile(queryPath, JSON.stringify({
      one: { visible: "A.png", query: "the red object" },
      two: { visible: "a.png", query: "the blue object" },
    }));

    const handlers = new Map();
    const tools = new Map();
    let activeTools = [];
    createGroundingSafetyExtension({
      cwd: root,
      sessionId: `image-cache-${archived ? "archive" : "file"}`,
      ...(archived ? { imageArchives: { visible: archivePath } } : {}),
    }).factory({
      on: (name, handler) => handlers.set(name, handler),
      registerTool: (tool) => tools.set(tool.name, tool),
      getActiveTools: () => activeTools,
      setActiveTools: (names) => { activeTools = names; },
      sendMessage() {},
    });
    await handlers.get("before_agent_start")({
      prompt: `Process 2 grounding records in ${queryPath}`,
      systemPromptOptions: { sections: {} },
    });
    const call = (name, params, context) => tools.get(name).execute(name, params, undefined, undefined, context);
    const context = { ui: { custom: async (factory) => {
      const details = factory({}, {}, {}, () => {}).groundingReview;
      return {
        type: "grounding_review_response", action: "confirm", bbox: details.bbox,
        status: "ok", confidence: .9, targetFound: true, candidateCount: 1, candidateRank: 1,
        reason: "The synthetic target is confirmed by the mocked human review.",
      };
    } } };

    for (const [key, expected] of [["one", [255, 0, 0]], ["two", [0, 0, 255]]]) {
      const loaded = await call("grounding_next_batch", { queryPath, outputDir });
      assert.equal(loaded.details.records[0].key, key);
      const image = loaded.content.find((block) => block.type === "image");
      const { data, info } = await sharp(Buffer.from(image.data, "base64")).raw().toBuffer({ resolveWithObject: true });
      const center = (Math.floor(info.height / 2) * info.width + Math.floor(info.width / 2)) * info.channels;
      assert.deepEqual([...data.subarray(center, center + 3)], expected);
      await call("grounding_save_result", {
        queryPath, outputDir, key, bbox: [.2, .2, .8, .8], status: "ok", confidence: .9,
        reason: "The synthetic target occupies the selected visible region.",
      }, context);
    }
  });
}
