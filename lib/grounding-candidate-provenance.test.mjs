import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";
const jiti = createJiti(import.meta.url);
const { assessGroundingCandidateProvenance } = await jiti.import("./grounding-candidate-provenance.ts");
const { GroundingViewRegistry } = await jiti.import("./grounding-views.ts");
const view = (modality = "visible", region = [0, 0, 1, 1]) => ({ modality, region, sourceWidth: 1000, sourceHeight: 800, width: 500, height: 400 });
const candidate = (id, measurementViewId, bbox = [.2, .2, .4, .4]) => ({ id, measurementViewId, bbox });

test("current-record visible overview and crop can share original source coordinates", () => {
  const registry = new GroundingViewRegistry();
  const overview = registry.register(view());
  const crop = registry.register(view("visible", [.1, .1, .5, .5]));
  const candidates = [candidate("a", overview.id), candidate("b", crop.id)];
  assert.deepEqual(assessGroundingCandidateProvenance(candidates, registry), []);
});

test("infrared/depth cannot supply visible geometry even with identical dimensions", () => {
  const registry = new GroundingViewRegistry();
  for (const modality of ["infrared", "depth"]) {
    const sensor = registry.register(view(modality));
    assert.equal(assessGroundingCandidateProvenance([candidate("sensor", sensor.id)], registry)[0].code, "non_visible_measurement_view");
  }
});

test("foreign or stale record IDs cannot resolve through another current-record registry", () => {
  const first = new GroundingViewRegistry();
  const second = new GroundingViewRegistry();
  const old = first.register(view());
  second.register(view());
  assert.equal(assessGroundingCandidateProvenance([candidate("old", old.id)], second)[0].code, "unavailable_measurement_view");
});

test("a crop must cover the entire candidate box, not just intersect or contain its center", () => {
  const registry = new GroundingViewRegistry();
  const crop = registry.register(view("visible", [.25, .25, .35, .35]));
  assert.equal(assessGroundingCandidateProvenance([candidate("wide", crop.id)], registry)[0].code, "measurement_view_coverage");
  assert.deepEqual(assessGroundingCandidateProvenance([candidate("edge", crop.id, [.25, .25, .35, .35])], registry), []);
});

test("inconsistent visible source dimensions fail closed rather than imply a transform", () => {
  const registry = new GroundingViewRegistry();
  const first = registry.register(view());
  const other = registry.register({ ...view(), sourceWidth: 2000 });
  assert.equal(assessGroundingCandidateProvenance([candidate("a", first.id), candidate("b", other.id)], registry)[0].code, "inconsistent_visible_source");
});
