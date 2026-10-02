// Offline visual inspection only. Draws the model proposal, never historical boxes.
import { readFile, readdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import sharp from "sharp";

const runDir = resolve(process.argv[2] ?? "");
if (!process.argv[2]) throw new Error("Usage: node e2e/grounding-benchmark-review.mjs RUN_DIRECTORY");
for (const entry of await readdir(runDir, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  const folder = join(runDir, entry.name);
  let result;
  try { result = JSON.parse(await readFile(join(folder, "result.json"), "utf8")); }
  catch (error) { if (error.code === "ENOENT") continue; throw error; }
  if (!result.prediction || !result.proposal) continue;
  const imageName = (await readdir(folder)).find((name) => /^review\.(?:jpg|png)$/u.test(name));
  if (!imageName) continue;
  const bytes = await readFile(join(folder, imageName));
  const { width, height } = await sharp(bytes).metadata();
  const [x1, y1, x2, y2] = result.prediction;
  const left = Math.max(0, Math.floor(x1 * width));
  const top = Math.max(0, Math.floor(y1 * height));
  const right = Math.min(width, Math.ceil(x2 * width));
  const bottom = Math.min(height, Math.ceil(y2 * height));
  const overlay = Buffer.from(`<svg width="${width}" height="${height}"><rect x="${x1 * width}" y="${y1 * height}" width="${(x2 - x1) * width}" height="${(y2 - y1) * height}" fill="none" stroke="#ffbd26" stroke-width="2"/></svg>`);
  await sharp(bytes).composite([{ input: overlay }]).png().toFile(join(folder, "proposal-full.png"));
  const cropWidth = Math.min(width, Math.max(180, (right - left) * 5));
  const cropHeight = Math.min(height, Math.max(140, (bottom - top) * 5));
  const cropLeft = Math.max(0, Math.min(width - cropWidth, Math.floor((left + right - cropWidth) / 2)));
  const cropTop = Math.max(0, Math.min(height - cropHeight, Math.floor((top + bottom - cropHeight) / 2)));
  const sourceCrop = { left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight };
  await sharp(bytes).extract(sourceCrop).resize({ width: 840, height: 720, fit: "inside" }).png().toFile(join(folder, "proposal-crop-clean.png"));
  const cropOverlay = Buffer.from(`<svg width="${cropWidth}" height="${cropHeight}"><rect x="${x1 * width - cropLeft}" y="${y1 * height - cropTop}" width="${(x2 - x1) * width}" height="${(y2 - y1) * height}" fill="none" stroke="#ffbd26" stroke-width="1"/></svg>`);
  await sharp(bytes).extract(sourceCrop).composite([{ input: cropOverlay }]).png().toBuffer()
    .then((crop) => sharp(crop).resize({ width: 840, height: 720, fit: "inside" }).png().toFile(join(folder, "proposal-crop.png")));
  await writeFile(join(folder, "visual-inspection.json"), JSON.stringify({ key: result.key, query: result.proposal.query,
    source: imageName, sourceWidth: width, sourceHeight: height, proposal: result.prediction, sourceCrop,
    note: "Yellow box is the unapproved model proposal. Clean crop retains source pixels without the box. No ground truth is used. Magnification is not new visual evidence." }, null, 2));
  console.log(`${result.key}: proposal-full.png, proposal-crop.png, proposal-crop-clean.png`);
}
