import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createJiti } from "jiti";
import sharp from "sharp";

const { createGroundingSafetyExtension } = await createJiti(import.meta.url).import("./grounding-safety-extension.ts");

test("crop caps display zoom, preserves exact source pixels and supports local inspection", async () => {
  const root = await mkdtemp(join(tmpdir(), "grounding-crop-regression-"));
  try {
    const source = join(root, "source");
    await mkdir(source);
    const queryPath = join(source, "queries.json");
    await writeFile(queryPath, JSON.stringify({ one: { visible: "one.png", query: "the black part" } }));
    await sharp({ create: { width: 1920, height: 1080, channels: 3, background: "#ffffff" } }).png().toFile(join(source, "one.png"));
    const handlers = new Map();
    const tools = new Map();
    let activeTools = [];
    createGroundingSafetyExtension({ cwd: root, sessionId: "crop-regression" }).factory({
      on(name, handler) { handlers.set(name, handler); },
      registerTool(tool) { tools.set(tool.name, tool); activeTools.push(tool.name); },
      getActiveTools: () => activeTools,
      setActiveTools(next) { activeTools = next; },
      sendMessage() {},
      sendUserMessage() {},
    });
    await handlers.get("before_agent_start")({ prompt: `grounding ${queryPath}`, systemPromptOptions: { sections: {} } });
    await tools.get("grounding_next_batch").execute("next", { queryPath, outputDir: join(root, "output"), limit: 1 });
    const view = tools.get("grounding_view");
    const base = { bbox: [0.62, 0.61, 0.69, 0.66], zoom: 3, region: [0.3, 0.55, 0.75, 0.85], reason: "Inspect the local black mark precisely" };
    const cropped = await view.execute("large", base);
    const payload = JSON.parse(cropped.content[1].text);
    assert.equal(payload.requestedZoom, 3);
    assert.equal(payload.zoomAdjusted, true);
    assert.equal(payload.maxDisplayLongSide, 1600);
    assert.deepEqual(payload.cropNormalized, base.region);
    assert.deepEqual(payload.cropPixels, [576, 594, 1440, 918]);
    assert.deepEqual(payload.displayedSizePixels, [1600, 600]);
    assert.equal(payload.magnification, 1600 / 864);
    assert.equal(payload.containsCurrentBbox, true);
    const image = cropped.content.find(block => block.type === "image");
    const metadata = await sharp(Buffer.from(image.data, "base64")).metadata();
    assert.deepEqual([metadata.width, metadata.height], [1600, 600]);

    const local = await view.execute("local", { ...base, region: [0.64001, 0.62501, 0.65501, 0.64001] });
    const localPayload = JSON.parse(local.content[1].text);
    assert.deepEqual(localPayload.currentBbox, base.bbox);
    assert.equal(localPayload.containsCurrentBbox, false);
    assert.equal(localPayload.intersectsCurrentBbox, true);
    assert.deepEqual(localPayload.currentBboxInCrop, [0, 0, 1, 1]);
    assert.deepEqual(localPayload.displayedNormalized, [1228 / 1920, 675 / 1080, 1258 / 1920, 692 / 1080]);
    assert.deepEqual(local.details.image.cropNormalized, localPayload.displayedNormalized);
    const disjoint = await view.execute("disjoint", { ...base, region: [0.1, 0.1, 0.2, 0.2] });
    const outside = JSON.parse(disjoint.content[1].text);
    assert.equal(outside.intersectsCurrentBbox, false);
    assert.equal(outside.currentBboxInCrop, null);
    assert.equal(outside.currentBboxIntersection, null);
    assert.equal(outside.overlay, "current_hypothesis_outside_crop");
    const outsideImage = disjoint.content.find(block => block.type === "image");
    const decoded = await sharp(Buffer.from(outsideImage.data, "base64")).removeAlpha().raw().toBuffer();
    let orangePixels = 0;
    for (let i = 0; i < decoded.length; i += 3) {
      if (decoded[i] > 180 && decoded[i + 1] > 90 && decoded[i + 1] < 210 && decoded[i + 2] < 70) orangePixels++;
    }
    assert.equal(orangePixels, 0, "a disjoint hypothesis must not invent an orange border box");
    for (const zoom of [0, -1, NaN, Infinity]) {
      await assert.rejects(view.execute("invalid", { ...base, zoom }), /finite number/);
    }
    await assert.rejects(view.execute("wrong-key", { ...base, key: "missing" }), /Load this record/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
