import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/HUANG LAB/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const phase='complete';const role='statistics';
const b='C:/Users/HUANG LAB/Documents/ChatGPT/IndividualParcellation/experiments/group_complete_revision_20261012/build';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const scene=JSON.parse(await fs.readFile(path.join(b,role==='statistics'?'scene.json':'rncut_scene.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}}),s=p.slides.add();s.background.fill='#FFFFFF';
const f=1280/scene.width;
function paint(c,a=1){return c==='none'?'none':a<1?`${c}/${Math.round(a*100)}`:c;}
for(const o of scene.objects){
 if(o.kind==='path'&&o.name==='//figure_1/patch_1/')continue;
 if(o.kind==='text'){
  const size=o.size*f,w=(o.text_width+1.6)*f,h=size*1.2,align=o.anchor==='middle'?'center':o.anchor==='end'?'right':'left';
  const x0=align==='center'?-w/2:align==='right'?-w:0,y0=-size*.91,t=o.rotation*Math.PI/180;
  const cx=o.x*f+Math.cos(t)*(x0+w/2)-Math.sin(t)*(y0+h/2),cy=o.y*f+Math.sin(t)*(x0+w/2)+Math.cos(t)*(y0+h/2);
  const q=s.shapes.add({name:'Group revision 20261012 '+o.name.slice(0,160),geometry:'textbox',position:{left:cx-w/2,top:cy-h/2,width:w,height:h,rotation:o.rotation},fill:'none',line:{fill:'none',width:0}});
  q.text=o.text;q.text.style={typeface:'Arial',fontSize:size,bold:o.bold,color:o.color,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};
 }else{
  if(o.dashed)throw Error('Statistical conditions may not be encoded by line style');
  const pts=o.commands.filter(c=>c.length>1);if(!pts.length)continue;
  let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];for(const c of pts){lo[0]=Math.min(lo[0],c[1]);lo[1]=Math.min(lo[1],c[2]);hi[0]=Math.max(hi[0],c[1]);hi[1]=Math.max(hi[1],c[2]);}
  const dx=Math.max(hi[0]-lo[0],1e-9),dy=Math.max(hi[1]-lo[1],1e-9);
  const commands=o.commands.map(c=>c[0]==='Z'?{close:{}}:{[c[0]==='M'?'moveTo':'lineTo']:{x:(c[1]-lo[0])/dx,y:(c[2]-lo[1])/dy}});
  s.shapes.add({name:'Group revision 20261012 '+o.name.slice(0,160),geometry:'custom',position:{left:lo[0]*f,top:lo[1]*f,width:Math.max(dx*f,.01),height:Math.max(dy*f,.01)},fill:paint(o.fill,o.fill_opacity),line:{fill:paint(o.stroke,o.stroke_opacity),width:o.stroke==='none'?0:o.stroke_width*f},customPaths:[{width:1,height:1,commands}]});
 }
}
s.speakerNotes.textFrame.setText(scene.caption);
await(await PresentationFile.exportPptx(p)).save(path.join(b,`native_${role}.pptx`));
console.log(JSON.stringify({phase,role,objects:s.shapes.items.length}));
