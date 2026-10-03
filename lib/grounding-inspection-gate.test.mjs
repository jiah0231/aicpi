import assert from 'node:assert/strict';
import test from 'node:test';
import { createJiti } from 'jiti';
const { redundantGroundingRegions, groundingInspectionPixels } = await createJiti(import.meta.url).import('./grounding-inspection-gate.ts');
const view = { id: 'actual-view', modality: 'visible', region: [.1,.1,.6,.6], sourceWidth: 1000, sourceHeight: 500, width: 1000, height: 500 };
const match = (region, previous = [view], modality = 'visible') => redundantGroundingRegions([{modality, region}], previous)[0];
test('canonical crop edges agree at floating point boundaries', () => {
 assert.deepEqual(groundingInspectionPixels([.1,.1,.600000000000001,.6],1000,500), [100,50,600,300]);
});
test('contained crops and near-identical jitter reuse already resolved pixels', () => {
 assert.equal(match([.2,.2,.4,.4]).id, view.id);
 assert.equal(match([.101,.1,.601,.6]).id, view.id);
 assert.equal(match([.7,.1,.9,.6]), undefined);
 assert.equal(match([.1,.1,.6,.6], [view], 'infrared'), undefined);
});
test('downsampled or derived images never gate higher-detail inspection', () => {
 assert.equal(match([.2,.2,.4,.4], [{...view,width:100,height:50}]), undefined);
 assert.equal(match([.2,.2,.4,.4], [{...view,derived:{kind:'image_processing'}}]), undefined);
});
test('comparison order and tool identity do not affect source reuse', () => {
 const requests = [{modality:'visible',region:[.2,.2,.4,.4]}, {modality:'visible',region:[.3,.3,.5,.5]}];
 assert.equal(redundantGroundingRegions(requests,[view]).filter(Boolean).length,2);
 assert.equal(redundantGroundingRegions(requests.reverse(),[view]).filter(Boolean).length,2);
});
test('capped large exact rerenders stop but higher-resolution subcrops remain available', () => {
 const large = {...view,region:[0,0,1,1],sourceWidth:4000,sourceHeight:4000,width:1600,height:1600};
 assert.equal(match([0,0,1,1],[large]).id,large.id);
 assert.equal(match([.2,.2,.4,.4],[large]),undefined);
});
test('comparison panel achievable resolution gates repeated panels, not finer single views', () => {
 const panel = {...view,region:[0,0,1,1],sourceWidth:4000,sourceHeight:4000,width:1600,height:992,displayRect:[16,400,556,940]};
 assert.equal(redundantGroundingRegions([{modality:'visible',region:[0,0,1,1],displayBounds:[1568,540]}],[panel])[0].id,panel.id);
 assert.equal(match([0,0,1,1],[panel]),undefined);
});
test('one-pixel thin strips still permit genuinely higher long-axis resolution', () => {
 const low = {...view,region:[0,0,1,1],sourceWidth:1000,sourceHeight:1000,width:100,height:100};
 assert.equal(match([0,0,1,.001],[low]),undefined);
 assert.equal(match([0,0,.001,1],[low]),undefined);
});
