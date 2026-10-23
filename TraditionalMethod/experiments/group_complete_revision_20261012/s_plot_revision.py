"""Editable group plate with20 genuine maps and all60 planned tests.

Structural adaptation of the established Python figure workflow. Preserve the
latest supplied source and16 existing surface groups. The unit is50 heldout people,
not vertices. K400 RNCut realizes195 left labels versus200 in other methods.
"""
from pathlib import Path
import csv,hashlib,json,sys,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from lxml import etree as ET
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import wilcoxon
WORK=Path(__file__).resolve().parents[2]
ROOT=WORK/'reproduction/nature_figures_20261007'
B=Path(__file__).resolve().parent/'build'
E=B/'evidence'
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(WORK/'reproduction/figure_skills/nature-skills/skills/nature-figure/scripts'))
import editable_scene as ES
import layout_boxplots as L
import results_figures as RF
from audit_panel_alignment import require_matplotlib_panel_alignment

METHODS=['Schaefer','NMF','DRFC','RNCut','CoBRA']
KS=[100,200,300,400]
COLORS=['#0072B2','#D55E00','#009E73','#AA4499']
METRICS=[('homogeneity','Homogeneity','H ↑'),('DCBC_x100','DCBC','DCBC × 100 ↑'),
         ('cosine_silhouette','FC-profile silhouette','Silhouette ↑')]
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def table(name,rows):
    with (B/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    gate=json.loads((E/'complete.json').read_text())
    assert gate['state']=='complete_user_authorized_all_maps_revision' and gate['participants']==50
    assert gate['maps']==35 and gate['observed_tests']==60 and gate['pending_tests']==0
    assert gate['accepted_measured_comparison_complete'] and not gate['K300_pending']
    manifest=json.loads((E/'manifest.json').read_text())
    for name,a in manifest.items():
        assert sha(E/name)==a['sha256'] and (E/name).stat().st_size==a['bytes']
    assert json.loads((E/'native_equivalence.json').read_text())['passed']
    source=json.loads((E/'summary.json').read_text());maps=dict(np.load(E/'atlas_labels_full_L.npz'))
    mask=np.load(E/'evaluation_mask_L.npy');assert mask.sum()==29251
    tests=source['comparisons'];assert len(tests)==60 and len(source['pending_comparisons'])==0
    subjects=[p['subject'] for p in source['participants']];assert len(set(subjects))==50
    # Independent recheck of every raw test and the complete60-test Holm family.
    for t in tests:
        d=np.array([p['metrics'][f'CoBRA_{t["K"]}'][t['metric']]-p['metrics'][f'{t["comparator"]}_{t["K"]}'][t['metric']] for p in source['participants']])
        p=float(wilcoxon(d,alternative='two-sided',zero_method='wilcox',method='approx',correction=False).pvalue) if np.any(d) else 1.
        assert abs(p-t['p_raw'])<1e-14
    p_all=np.array([t['p_raw'] for t in tests])
    idx=np.argsort(p_all,kind='stable');adj=np.minimum(1,np.maximum.accumulate(p_all[idx]*(60-np.arange(60))))
    check=np.empty(60);check[idx]=adj
    assert np.allclose(check,[t['p_holm'] for t in tests],atol=1e-14,rtol=0)
    K300_evaluation=source['protocol']['accepted_revision']['K300_evaluation_left']
    data=[];summary=[];topology=[]
    edges=np.load(B/'edges_L.npy');n=len(mask)
    graph=csr_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n));graph=graph+graph.T
    for method in METHODS:
        for k in KS:
            key=f'{method}_{k}'
            assert key in maps,key
            v=maps[key];actual=len(np.unique(v[v>=0]));evaluated=len(np.unique(v[mask]))
            assert actual==(195 if key=='RNCut_400' else k//2)
            # The frozen common mask excludes small native RNCut parcels.
            # Record the effective count; never restore excluded vertices.
            assert evaluated==({'RNCut_200':99,'RNCut_300':K300_evaluation,'RNCut_400':192}.get(key,actual))
            counts=[];largest=[]
            for label in np.unique(v[v>=0]):
                ids=np.flatnonzero(v==label);cc,c=connected_components(graph[ids][:,ids],directed=False)
                counts.append(cc);largest.append(np.bincount(c).max())
            topology.append(dict(map=key,K_requested_bilateral=k,K_left_actual=actual,
                K_left_evaluated=evaluated,disconnected_parcel_fraction=float(np.mean(np.array(counts)>1)),extra_components=int(sum(counts)-actual),
                largest_component_vertex_fraction=float(sum(largest)/(v>=0).sum())))
            for metric,_,_ in METRICS:
                a=np.array([p['metrics'][key][metric] for p in source['participants']]);assert np.isfinite(a).all()
                for p,val in zip(source['participants'],a):data.append(dict(subject=p['subject'],map=key,family=method,
                     K_bilateral_requested=k,K_left_native=actual,K_left_evaluated=evaluated,metric=metric,value=float(val)))
                summary.append(dict(method=method,K_bilateral_requested=k,K_left_actual=actual,metric=metric,n=50,
                     mean=float(a.mean()),SD=float(a.std(ddof=1)),median=float(np.median(a)),Q1=float(np.quantile(a,.25)),
                     Q3=float(np.quantile(a,.75)),min=float(a.min()),max=float(a.max())))
    assert len(data)==3000 and len(summary)==60 and len(topology)==20
    table('source_data.csv',data);table('summary.csv',summary);table('group_topology.csv',topology)
    table('paired_tests_family60.csv',tests)
    shown=[t for t in tests if t['comparator'] in METHODS];assert len(shown)==48
    table('paired_tests_displayed.csv',shown)
    (B/'pending_tests.json').write_text(json.dumps(source['pending_comparisons'],indent=2))
    contract=dict(question='How does heldout functional quality vary across the displayed group methods and requested resolutions?',
         claim='Quality is metric dependent. All requested maps are measured; RNCut K400 realizes195 left parcels.',
         archetype='image plate + quant',backend='python',dimensions_mm=[338.666667,190.5],minimum_font_pt=5,
         source_template_sha256=sha(B/'user_source.pptx'),participants=50,displayed_maps=20,observations=3000,
         missing=[],planned_tests=60,observed_tests=60,statistics='Paired two-sided Wilcoxon with exact Holm adjustment over all60 observed planned tests.',
         full_original_campaign_complete=False,actual_K_disclosed=True,brain_geometry_preserved=16)
    (B/'figure_contract.json').write_text(json.dumps(contract,indent=2))
    plt.rcParams.update({'font.family':'Arial','font.size':8.5,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.labelsize':8.2,'ytick.labelsize':8.2,'legend.frameon':False})
    W,H=1280,720;fig=plt.figure(figsize=(W/96,H/96));axes=[]
    handles=[Line2D([],[],linestyle='none',marker='s',markersize=5.7,color=c,label=f'K = {k}') for k,c in zip(KS,COLORS)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(705/W,1-145/H),ncol=4,
        handletextpad=.35,columnspacing=1,borderaxespad=0,fontsize=8.4)
    offsets=[-.30,-.10,.10,.30]
    for j,(metric,title,ylabel) in enumerate(METRICS):
        top=200+j*144
        fig.text(682/W,1-(178+j*144)/H,f'{chr(66+j)}   {title}',fontsize=9.7,fontweight='bold',va='top',color='#202A32')
        ax=fig.add_axes([715/W,1-(top+80)/H,450/W,80/H]);axes.append(ax)
        ax.set_xlim(-.6,4.6);ax.set_xticks(range(5));ax.set_xticklabels(METHODS,rotation=32,ha='right',rotation_mode='anchor')
        ax.tick_params(axis='both',length=2.2,pad=2);values=[]
        for mi,m in enumerate(METHODS):
            for ki,k in enumerate(KS):
                key=f'{m}_{k}'
                assert key in maps,key
                a=np.array([p['metrics'][key][metric] for p in source['participants']]);values.extend(a)
                c=COLORS[ki]
                ax.boxplot([a],positions=[mi+offsets[ki]],widths=.14,manage_ticks=False,whis=1.5,patch_artist=True,
                    boxprops={'facecolor':c,'edgecolor':c,'linewidth':.6},medianprops={'color':'#17232C','linewidth':.8},
                    whiskerprops={'color':c,'linewidth':.6},capprops={'color':c,'linewidth':.6},
                    flierprops={'marker':'o','markersize':1.4,'markerfacecolor':c,'markeredgecolor':c,'markeredgewidth':.3})
        lo,hi=min(values),max(values);span=hi-lo;ax.set_ylim(lo-.07*span,hi+.42*span)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3));ax.set_ylabel(ylabel,fontsize=8.5,labelpad=2)
        for mi,m in enumerate(METHODS[:-1]):
            for ki,k in enumerate(KS):
                match=[t for t in shown if t['comparator']==m and t['K']==k and t['metric']==metric]
                assert len(match)==1
                p=match[0]['p_holm']
                star='****' if p<.0001 else '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
                ax.text(mi+offsets[ki],hi+.18*span,star,ha='center',va='center',fontsize=6.2,color=COLORS[ki])
    # Match the empty RNCut row to the supplied grid without moving any brain.
    with zipfile.ZipFile(B/'user_source.pptx') as z:
        tree=ET.fromstring(z.read('ppt/slides/slide4.xml')).find('p:cSld/p:spTree',NS)
        groups=[g for g in tree if g.tag.endswith('}grpSp')]
        assert len(groups)==16
        boxes=[]
        for g in groups:
            t=g.find('p:grpSpPr/a:xfrm',NS);a=t.find('a:off',NS);e=t.find('a:ext',NS)
            boxes.append([int(a.get('x'))/9525,int(a.get('y'))/9525,int(e.get('cx'))/9525,int(e.get('cy'))/9525])
        rows=sorted(boxes,key=lambda a:a[1]);drfc=sorted(rows[8:12],key=lambda a:a[0]);cobra=sorted(rows[12:16],key=lambda a:a[0])
        slots=[[a[0],(a[1]+b[1])/2,a[2],a[3]] for a,b in zip(drfc,cobra)]
    foot=[
        '50 held-out participants. Nominal bilateral K = 100 / 200 / 300 / 400 → left K = 50 / 100 / 150 / 200. RNCut (surface adaptation): K400 realizes 195.',
        f'Same 29,251-vertex mask; RNCut evaluated left K: 50 / 99 / {K300_evaluation} / 192. Training: NMF, DRFC, RNCut 500; CoBRA 450. Filter: 0.01–0.1 Hz.',
        'Boxes: median, IQR, 1.5-IQR whiskers, outliers. Tests vs CoBRA: Holm over all 60 observed planned tests. * < .05, ** < .01, *** < .001, **** < .0001.'
    ]
    for i,line in enumerate(foot):fig.text(153/W,1-(630+i*15)/H,line,fontsize=6.2,color='#58646F',va='top')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=['H','DCBC','silhouette'],row_groups=[],
        column_groups=[['H','DCBC','silhouette']],json_out=B/'alignment.json',overlay_svg=B/'alignment.svg',tolerance_pt=1.5,strict=True)
    fig.savefig(B/'statistics.svg');fig.savefig(B/'statistics.pdf');fig.savefig(B/'statistics.png',dpi=300);plt.close(fig)
    scene=ES.svg_scene('Complete accepted group comparison',[],svg_path=B/'statistics.svg')
    ES.BUILD=B/'brain_paths';ES.BUILD.mkdir(exist_ok=True);ES.brain_cache.clear()
    provenance=[]
    for ki,k in enumerate(KS):
        key=L.aligned(maps[f'RNCut_{k}'],maps[f'CoBRA_{k}'],f'accepted_RNCut_{k}',k);RF.D=ES.D=L.D
        # SVG uses points, whereas supplied slide slots use CSS pixels.
        objects=ES.brain_paths(key,k,[v*.75 for v in slots[ki]])
        scene['objects'].extend(objects)
        provenance.append(dict(K_requested=k,K_left_actual=195 if k==400 else k//2,
            source_label_sha256=hashlib.sha256(maps[f'RNCut_{k}'].tobytes()).hexdigest(),
            bbox_css_px=slots[ki],display_only_permutation=L.PERM[-1],labels_repaired=False))
    scene['caption']=('Group-level functional parcellation. Schaefer, NMF32 plus K-means, MICCAI2026/DRFC, RNCut surface adaptation and CoBRA. '
        'Twenty genuine maps and3000 observations on50 heldout participants. All four requested resolutions have maps and measured comparisons. '
        'Requested bilateralK100/200/300/400 means left50/100/150/200 parcels, except native RNCut K400 realizes195, explicitly disclosed. '
        'That comparison is at matching requested resolution, not matching realized parcel count. Original author output is preserved without repair. '
        'All measurements use29251 common vertices,0.01–0.1Hz filtering and unchanged metric code. The common mask excludes small RNCut parcels: '
        f'native/evaluated left counts are50/50,100/99,150/{K300_evaluation} and195/192 for requestedK100,200,300 and400. No excluded vertices are restored. '
        'K300 uses the disclosed fixed-seed training-only rounding fallback, with the unchanged original author optimizer. '
        'Training differs:500 for NMF/DRFC/RNCut,450 for CoBRA. '
        'Colors encode requested resolution and match the supplied individual figure. Boxes:median,IQR,1.5-IQR whiskers,all outliers. '
        'Paired two-sided Wilcoxon versus CoBRA,48 displayed tests among all60 planned tests; exact Holm adjustment over the complete family. '
        'Star thresholds:0.05,0.01,0.001,0.0001. '
        'Original16 non-RNCut surface groups, spacing and all other slides remain unchanged. Display-only Hungarian color permutation matches the reference. '
        'No universal superiority claim. This completes the user-approved measured Group comparison; the strict original matched-realized-count campaign remains incomplete because of K400.')
    (B/'scene.json').write_text(json.dumps(scene,separators=(',',':')),encoding='utf-8')
    (B/'display_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(dict(maps=20,observations=len(data),native_objects=len(scene['objects']),tests=48)))


if __name__=='__main__':main()
