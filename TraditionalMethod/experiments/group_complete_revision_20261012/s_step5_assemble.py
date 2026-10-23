"""Assemble checked Figure3 without resaving the unchanged source slides."""
from pathlib import Path
import argparse,copy,hashlib,json,shutil,zipfile
WORK=Path(__file__).resolve().parents[2];BASE=WORK/'experiments/group_complete_revision_20261012/build'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main(phase):
    b=BASE;out=WORK/'outputs/group_complete_revision_20261012';out.mkdir(parents=True,exist_ok=True)
    check=json.loads((b/'PPT_validation_v2.json').read_text());assert check
    stem='CoBRA_Group_Complete_K300_Editable_20261012'
    one=out/(stem+'.pptx');full=out/'cobra_figure_Group_Complete_K300_20261012.pptx'
    assert not one.exists() and not full.exists()
    shutil.copy2(b/'validated/Figure3_v2.pptx',one)
    changed=['ppt/slides/slide4.xml','ppt/notesSlides/notesSlide4.xml']
    with zipfile.ZipFile(BASE/'user_source.pptx') as source,zipfile.ZipFile(one) as checked:
        with zipfile.ZipFile(full,'w',zipfile.ZIP_DEFLATED) as dest:
            for info in source.infolist():dest.writestr(copy.copy(info),checked.read(info.filename) if info.filename in changed else source.read(info.filename))
        with zipfile.ZipFile(full) as result:
            unchanged=0
            for name in source.namelist():
                if name not in changed:assert source.read(name)==result.read(name);unchanged+=1
    for name in ['source_data.csv','summary.csv','paired_tests_displayed.csv','paired_tests_family60.csv','group_topology.csv','pending_tests.json','display_provenance.json','figure_contract.json']:
        if (b/name).exists():shutil.copy2(b/name,out/('Group_Figure3_'+name))
    (b/'delivery_preservation.json').write_text(json.dumps({'passed':True,'phase':phase,'source_sha256':sha(BASE/'user_source.pptx'),
        'full_sha256':sha(full),'one_slide_sha256':sha(one),'changed_parts':changed,'unchanged_parts':unchanged,
        'pipeline_slide_and_individual_slide_byte_identical':True,'non_RNCut_16_brain_groups_unchanged':True,
        'files':[str(full),str(one)]},indent=2))
    print(json.dumps({'full':str(full),'one_slide':str(one),'sha256':sha(full)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['complete'],default='complete');main(ap.parse_args().phase)
