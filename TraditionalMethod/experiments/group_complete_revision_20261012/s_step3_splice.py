"""Insert editable measured statistics into Figure3, preserving the supplied deck."""
from pathlib import Path
import argparse,copy,hashlib,json,posixpath,zipfile
from lxml import etree as E

WORK=Path(__file__).resolve().parents[2]
BASE=WORK/'experiments/group_complete_revision_20261012/build'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
PART='ppt/slides/slide4.xml'
RN_IDS={'2302','2299','2292','2289'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canonical=lambda e:E.tostring(e,method='c14n')
serialize=lambda e:E.tostring(e,encoding='utf-8',xml_declaration=True,standalone=True)

def closed_scope(source,output):
    """Normal bounded-part validation of the edited slide; retain source XML."""
    with zipfile.ZipFile(source) as z:
        updates={}
        pres=E.fromstring(z.read('ppt/presentation.xml'));lst=pres.find('p:sldIdLst',NS);keep=lst[3]
        rid=keep.get('{'+NS['r']+'}id')
        for el in list(lst):
            if el is not keep:lst.remove(el)
        updates['ppt/presentation.xml']=serialize(pres)
        rels=E.fromstring(z.read('ppt/_rels/presentation.xml.rels'))
        for el in list(rels):
            if el.get('Type','').endswith('/slide') and el.get('Id')!=rid:rels.remove(el)
        updates['ppt/_rels/presentation.xml.rels']=serialize(rels)
        names=set(z.namelist());reachable=set();stack=['']
        while stack:
            part=stack.pop()
            if part:
                if part in reachable:continue
                assert part in names,part
                reachable.add(part)
            rp=posixpath.join(posixpath.dirname(part),'_rels',posixpath.basename(part)+'.rels') if part else '_rels/.rels'
            if rp not in names:continue
            reachable.add(rp)
            for el in E.fromstring(updates.get(rp,z.read(rp))):
                if el.get('TargetMode')=='External':continue
                target=el.get('Target');target=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(part),target))
                stack.append(target)
        types=E.fromstring(z.read('[Content_Types].xml'))
        for el in list(types):
            if el.tag.endswith('Override') and el.get('PartName').lstrip('/') not in reachable:types.remove(el)
        updates['[Content_Types].xml']=serialize(types);reachable.add('[Content_Types].xml')
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as out:
            for info in z.infolist():
                if info.filename in reachable:out.writestr(copy.copy(info),updates.get(info.filename,z.read(info.filename)))
    return {'path':str(output),'sha256':sha(output),'parts':len(reachable)}

def main(phase):
    b=BASE;source=BASE/'user_source.pptx';scene=json.loads((b/'scene.json').read_text())
    with zipfile.ZipFile(source) as original,zipfile.ZipFile(b/'native_statistics.pptx') as native:
        old=E.fromstring(original.read(PART));new=copy.deepcopy(old);tree=new.find('p:cSld/p:spTree',NS)
        kept=[];removed=[]
        for el in list(tree):
            n=el.find('.//p:cNvPr',NS)
            if n is not None and n.get('name','').startswith(('Group Figure3 statistics','Group revision 20261012 ')):
                tree.remove(el);removed.append(n.get('id'))
            elif el.tag.endswith('}grpSp'):kept.append(n.get('id'))
        assert len(removed)>300 and len(kept)==16
        start=max(int(n.get('id')) for n in new.findall('.//p:cNvPr',NS))+1;count=0
        add=E.fromstring(native.read('ppt/slides/slide1.xml')).find('p:cSld/p:spTree',NS)
        for el in add:
            if el.tag.rsplit('}',1)[-1] in ['nvGrpSpPr','grpSpPr']:continue
            q=copy.deepcopy(el);assert not q.findall('.//a:blip',NS)
            for n in q.findall('.//p:cNvPr',NS):n.set('id',str(start));start+=1
            tree.append(q);count+=1
        for ident in kept:
            a=next(el for el in old.find('p:cSld/p:spTree',NS) if el.find('.//p:cNvPr',NS) is not None and el.find('.//p:cNvPr',NS).get('id')==ident)
            c=next(el for el in tree if el.find('.//p:cNvPr',NS) is not None and el.find('.//p:cNvPr',NS).get('id')==ident)
            assert canonical(a)==canonical(c),f'Brain map {ident} changed'
        updates={PART:serialize(new)}
        note_part='ppt/notesSlides/notesSlide4.xml'
        note=E.fromstring(original.read(note_part))
        for sp in note.findall('.//p:sp',NS):
            ph=sp.find('p:nvSpPr/p:nvPr/p:ph',NS)
            if ph is not None and ph.get('type')=='body':
                body=sp.find('p:txBody',NS)
                for child in list(body):
                    if child.tag=='{'+NS['a']+'}p':body.remove(child)
                para=E.SubElement(body,'{'+NS['a']+'}p');run=E.SubElement(para,'{'+NS['a']+'}r');E.SubElement(run,'{'+NS['a']+'}t').text=scene['caption']
        updates[note_part]=serialize(note)
        candidate=b/'candidate_v2.pptx'
        with zipfile.ZipFile(candidate,'w',zipfile.ZIP_DEFLATED) as out:
            for info in original.infolist():out.writestr(copy.copy(info),updates.get(info.filename,original.read(info.filename)))
        with zipfile.ZipFile(candidate) as check:
            for name in original.namelist():
                if name not in updates:assert check.read(name)==original.read(name)
        receipt={'passed':True,'source_sha256':sha(source),'phase':phase,'changed_parts':list(updates),
            'all_other_slides_byte_identical':True,'preserved_brain_groups':16,'brain_geometry_and_spacing_unchanged':True,
            'old_statistics_shapes_replaced':len(removed),'native_objects_inserted':count}
        (b/'preservation_audit.json').write_text(json.dumps(receipt,indent=2))
    scopes=[closed_scope(source,b/'source_Figure3_scope.pptx'),closed_scope(candidate,b/'candidate_Figure3_scope_v2.pptx')]
    (b/'validation_scope.json').write_text(json.dumps(scopes,indent=2));print(json.dumps(receipt))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['complete'],default='complete');main(ap.parse_args().phase)
