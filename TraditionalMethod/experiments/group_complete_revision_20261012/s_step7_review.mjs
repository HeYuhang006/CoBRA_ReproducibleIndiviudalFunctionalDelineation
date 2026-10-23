import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
const work='C:/Users/HUANG LAB/Documents/ChatGPT/IndividualParcellation';
const runtime='C:/Users/HUANG LAB/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const out=path.join(work,'outputs/group_complete_revision_20261012');
const b=path.join(work,'experiments/group_complete_revision_20261012/build');
const {PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const full=await PresentationFile.importPptx(await FileBlob.load(path.join(out,'cobra_figure_Group_Complete_K300_20261012.pptx')));
if(full.slides.items.length!==7)throw Error('Latest supplied deck has seven slides');
await fs.mkdir(path.join(b,'review'),{recursive:true});
let groupHash;const previews=[];
for(let i=0;i<full.slides.items.length;i++){
 const slide=full.slides.items[i];
 const png=Buffer.from(await (await full.export({slide,format:'png',scale:1.5})).arrayBuffer());
 const target=path.join(b,'review',`slide${i+1}.png`);await fs.writeFile(target,png);
 previews.push({slide:i+1,path:target,sha256:sha(png)});
 if(i===3){
  groupHash=sha(png);await fs.writeFile(path.join(out,'Group_Complete_Preview_20261012.png'),png);
  await fs.writeFile(path.join(b,'review/group.layout.json'),await (await slide.export({format:'layout'})).text());
 }
}
const one=await PresentationFile.importPptx(await FileBlob.load(path.join(out,'CoBRA_Group_Complete_K300_Editable_20261012.pptx')));
if(one.slides.items.length!==1)throw Error('Standalone Group must have one slide');
const png=Buffer.from(await (await one.export({slide:one.slides.items[0],format:'png',scale:1.5})).arrayBuffer());
await fs.writeFile(path.join(b,'review/group_standalone.png'),png);
if(sha(png)!==groupHash)throw Error('Standalone Group rendering differs from full deck');
await fs.writeFile(path.join(b,'render_receipt.json'),JSON.stringify({actual_PPT_reimported:true,slides:7,previews,standalone_matches_full:true,visual_review_complete:false},null,2));
console.log(JSON.stringify({actual_PPT_reimported:true,slides:7,standalone_matches_full:true,review_folder:path.join(b,'review')}));
