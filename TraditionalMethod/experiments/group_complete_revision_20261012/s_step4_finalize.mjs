import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const work='C:/Users/HUANG LAB/Documents/ChatGPT/IndividualParcellation';
const runtime='C:/Users/HUANG LAB/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill='C:/Users/HUANG LAB/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations';
const phase='complete';
const b=path.join(work,'experiments/group_complete_revision_20261012/build');
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const scopes=JSON.parse(await fs.readFile(path.join(b,'validation_scope.json'),'utf8'));
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(path.join(b,'validated'),{recursive:true});
await finalizePresentation({workspaceDir:work,candidatePath:path.join(b,'candidate_Figure3_scope_v2.pptx'),
 finalPath:path.join(b,'validated/Figure3_v2.pptx'),pythonExecutable:path.join(runtime,'python/python.exe'),
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['Arial'],referencePath:path.join(b,'source_Figure3_scope.pptx'),referenceSha256:scopes[0].sha256},
 verifyArtifactToolImport:true,receiptPath:path.join(b,'PPT_validation_v2.json')});
console.log(JSON.stringify({phase,validated:'Figure3.pptx'}));
