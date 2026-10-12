"""Editable PowerPoint: native parcel polygons, text, boxes, whiskers and points.

Requires completed fresh comparisons. No screenshot, PNG or PDF is embedded.
The inflated left hemisphere is orthographically projected; surfaces use actual
triangle labels. Display permutations are fixed from Day1, never refit by visit.
"""
from pathlib import Path
import csv
import numpy as np
from f_runtime import CODE,f_read,f_save,f_import

def f_table(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))

def f_text(slide,text,x,y,w,h,size=10,color='252525'):
    from pptx.util import Inches,Pt
    from pptx.dml.color import RGBColor
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame;tf.word_wrap=True
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
    p=tf.paragraphs[0];p.text=str(text);p.font.name='Arial';p.font.size=Pt(size);p.font.color.rgb=RGBColor.from_string(color)
    return box

def f_line(slide,x1,y1,x2,y2,color='333333',width=.6):
    from pptx.util import Inches,Pt
    from pptx.dml.color import RGBColor
    shape=slide.shapes.add_connector(1,Inches(x1),Inches(y1),Inches(x2),Inches(y2));shape.line.color.rgb=RGBColor.from_string(color);shape.line.width=Pt(width)

def f_rect(slide,x,y,w,h,color,outline=None):
    from pptx.util import Inches
    from pptx.dml.color import RGBColor
    shape=slide.shapes.add_shape(1,Inches(x),Inches(y),Inches(w),Inches(h));shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(color)
    if outline:shape.line.color.rgb=RGBColor.from_string(outline)
    else:shape.line.fill.background()
    return shape

def f_surface(slide,labels,xyz,faces,vertices,rect,palette):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.geometry.polygon import orient
    from pptx.util import Inches
    from pptx.dml.color import RGBColor
    full=np.full(len(xyz),-1,dtype='int32');full[vertices]=labels
    f=full[faces];values=np.where(f[:,0]==f[:,1],f[:,0],np.where(f[:,1]==f[:,2],f[:,1],f[:,0]));values[(f<0).any(1)]=-1
    normal=np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]])
    visible=normal[:,0]<0;xy=xyz[:,[1,2]].copy();xy[:,1]*=-1
    low=xy.min(0);span=np.ptp(xy,axis=0);x,y,w,h=rect;s=min(w/span[0],h/span[1]);points=(xy-low)*s+np.array([x+(w-span[0]*s)/2,y+(h-span[1]*s)/2])
    for label in np.unique(values[visible]):
        polygons=[Polygon(points[tri]) for tri in faces[visible&(values==label)]]
        union=unary_union(polygons)
        for polygon in ([union] if union.geom_type=='Polygon' else list(union.geoms)):
            if polygon.area<1e-8:continue
            polygon=orient(polygon,sign=1.0)
            loops=[np.asarray(polygon.exterior.coords)]+[np.asarray(v.coords) for v in polygon.interiors]
            first=loops[0];builder=slide.shapes.build_freeform(Inches(first[0,0]),Inches(first[0,1]))
            builder.add_line_segments([(Inches(a),Inches(b)) for a,b in first[1:]],close=True)
            for loop in loops[1:]:
                builder.move_to(Inches(loop[0,0]),Inches(loop[0,1]));builder.add_line_segments([(Inches(a),Inches(b)) for a,b in loop[1:]],close=True)
            shape=builder.convert_to_shape();shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string('D0D3D5' if label<0 else palette[int(label)%len(palette)]);shape.line.fill.background()

def f_palette(k):
    from matplotlib.colors import hsv_to_rgb,to_hex
    return [to_hex(c,keep_alpha=False)[1:].upper() for c in hsv_to_rgb(np.c_[(np.arange(k)*.61803398875)%1,np.full(k,.63),np.full(k,.92)])]

def f_box(slide,values,x,y,w,h,color,limits):
    values=np.asarray(values);q1,med,q3=np.quantile(values,[.25,.5,.75]);iqr=q3-q1
    low=values[values>=q1-1.5*iqr].min();high=values[values<=q3+1.5*iqr].max()
    project=lambda v:y+h-h*(v-limits[0])/(limits[1]-limits[0])
    f_line(slide,x+w/2,project(low),x+w/2,project(high),color)
    for v in [low,high]:f_line(slide,x+.2*w,project(v),x+.8*w,project(v),color)
    f_rect(slide,x,project(q3),w,max(project(q1)-project(q3),.002),'FFFFFF',color)
    f_line(slide,x,project(med),x+w,project(med),color,1)

def f_individual_panel(slide,rows,tests,metric,rect):
    x,y,w,h=rect;colors=['9C79BA','DF9B52','648CB4','61A79B','D65454'];methods=['PCA','Wang','SpatialWard','GPIP','CoBRA']
    f_text(slide,metric,x,y-.27,w,.25,11)
    for side,K in enumerate([18,100]):
        subset=[r for r in rows if int(r['K'])==K and r['metric']==metric and (metric=='Fragments' or r['contrast']=='Day1_Retest')]
        values=[float(r['value']) for r in subset];limits=(0,1) if metric!='Fragments' else (0,max(1,max(values)*1.08))
        bx=x+side*w*.5;bw=w*.43;f_line(slide,bx,y,bx,y+h)
        for t in [0,.5,1]:
            value=limits[0]+t*(limits[1]-limits[0]);f_text(slide,f'{value:.2g}',bx-.25,y+h-t*h-.06,.25,.14,7)
        for j,(method,color) in enumerate(zip(methods,colors)):
            v=[float(r['value']) for r in subset if r['method']==method]
            if not v:raise RuntimeError('Missing box data: '+str((K,metric,method)))
            pos=bx+.03+j*bw/5;f_box(slide,v,pos,y,bw/5*.55,h,color,limits)
        f_text(slide,'K='+str(K),bx,y+h+.06,bw,.16,8)
    relevant=[r for r in tests if r['metric']==metric and (metric=='Fragments' or r['contrast']=='Day1_Retest')]
    # Explicit adjusted P values are editable text; same-subject tests, no fabricated stars.
    text='; '.join(f"{r['K']}:{r['comparator']} pH={float(r['p_holm']):.2g}" for r in relevant)
    f_text(slide,text,x,y+h+.26,w,.4,6)

def f_figures(c,w):
    import nibabel as nib
    from pptx import Presentation
    from pptx.util import Inches
    from f_comparisons import f_load_maps
    I=w/'reports/individual';G=w/'reports/group'
    assert f_read(I/'completion.json')['complete'],'Finish all individual comparisons before final figure'
    rows=f_table(I/'subject_metrics.csv');tests=f_table(I/'paired_tests.csv');group=f_table(G/'subject_metrics.csv')
    mp=np.load(w/'runs/pilot32_MSMAll_samplez/mapping.npz');vertices=mp['L_vertices']
    mesh=nib.load(Path(c['template_root'])/'tpl-fsLR_den-32k_hemi-L_inflated.surf.gii');xyz=mesh.agg_data('pointset');faces=mesh.agg_data('triangle')
    ids=[r['subject'] for r in f_read(w/'runs/connectome_validation_20261007_v1/retest/subjects.json')];subject=ids[0]
    prs=Presentation();prs.slide_width=Inches(16);prs.slide_height=Inches(8)
    slide=prs.slides.add_slide(prs.slide_layouts[6]);methods=['PCA','Wang','SpatialWard','GPIP','CoBRA'];visits=['Day1','Day2','Retest']
    SS=f_import('final/plotting.py');N=w/'runs/coarse_network18_20261008_v1';m=np.load(N/'geometry/evaluation_mask_L.npy');nearest=np.load(N/'geometry/nearest4_L.npy')
    display=[]
    for side,K in enumerate([18,100]):
        x0=.7 if side==0 else 10.0
        for j,v in enumerate(visits):f_text(slide,v,x0+j*1.78,.25,1.65,.3,14)
        prior=np.load(N/'geometry/author_labels_L.npy')[m] if K==18 else np.load(w/'runs/corrected_personal_multires_20261006_v1/priors/L/50/labels.npy')
        for i,method in enumerate(methods):
            maps=[f_load_maps(w,K,method,subject,v,0) for v in visits]
            if K==18:
                n=int(m.sum());maps=[v[:n] for v in maps]
            k=18 if K==18 else 50
            from scipy.optimize import linear_sum_assignment
            overlap=np.bincount(maps[0].astype('int64')*k+prior,minlength=k*k).reshape(k,k);a,b=linear_sum_assignment(-overlap);perm=np.arange(k);perm[a]=b
            display.append(dict(K=K,method=method,fixed_Day1_permutation=perm.tolist()))
            if side==0:f_text(slide,method,.02,1.0+i*1.26,.65,.35,10)
            for j,v in enumerate(maps):
                labels=perm[v]
                if K==18:
                    full=np.full(2562,-1,dtype='int32');full[m]=labels;labels=full[nearest]
                f_surface(slide,labels,xyz,faces,vertices,(x0+j*1.78,.7+i*1.26,1.7,1.2),f_palette(k))
        f_text(slide,'K='+str(K),x0+2,7.16,1.5,.3,14)
    for i,metric in enumerate(['ARI','Parcel_Dice','Fragments']):f_individual_panel(slide,rows,tests,metric,(6.36,.95+i*2.02,3.25,1.19))
    f_text(slide,'Retest interval follows HCP visit metadata. Wang100 is adapted diagnostic; Wang18 is later HFR release. 18: excess components over distributed prior; Ward is an areal control.',.1,7.62,15.8,.28,8)
    slide=prs.slides.add_slide(prs.slide_layouts[6]);groupmethods=['Schaefer','DRFC','CoBRA_raw','CoBRA']
    for j,K in enumerate(c['K_bilateral']):f_text(slide,'K='+str(K),.7+j*1.8,.25,1.6,.3,14)
    for i,method in enumerate(groupmethods):
        f_text(slide,method,.02,1+i*1.55,.7,.35,10)
        for j,K in enumerate(c['K_bilateral']):
            k=K//2
            if method=='Schaefer':lab=np.asarray(nib.load(CODE/f'resources/atlases/Schaefer2018_{K}Parcels_7Networks_order.dlabel.nii').dataobj).ravel()[mp['L_columns']]-1
            else:
                path=w/(f'runs/full500_MSMAll_samplez/labels_L_{k}.npy' if method=='DRFC' else f'runs/vae450_spatialweight_20261006_v1/labels_LPE1_L_{k}.npy' if method=='CoBRA_raw' else f'runs/corrected_personal_multires_20261006_v1/priors/L/{k}/labels.npy')
                lab=np.load(path)
            f_surface(slide,lab,xyz,faces,vertices,(.7+j*1.8,.72+i*1.55,1.7,1.4),f_palette(k))
    families=sorted(set(r['method'].rsplit('_',1)[0] for r in group));colors=f_palette(len(families));ks=sorted(set(int(r['K_bilateral']) for r in group))
    for panel,metric in enumerate(['homogeneity','DCBC_x100','FC_profile_silhouette']):
        x,y,pw,ph=8.6,.9+panel*2.15,6.8,1.4;subset=[r for r in group if r['metric']==metric];vals=[float(r['value']) for r in subset];lo,hi=min(vals),max(vals);pad=(hi-lo)*.08 or .01;lo-=pad;hi+=pad
        f_text(slide,metric,x,y-.28,pw,.25,13);f_line(slide,x,y,x,y+ph);f_line(slide,x,y+ph,x+pw,y+ph)
        for v in np.linspace(lo,hi,4):f_text(slide,f'{v:.3g}',x-.45,y+ph-ph*(v-lo)/(hi-lo)-.08,.43,.2,8)
        for name in sorted(set(r['method'] for r in subset)):
            part=[r for r in subset if r['method']==name];K=int(part[0]['K_bilateral']);values=np.array([float(r['value']) for r in part]);mean=values.mean();sd=values.std(ddof=1)
            family=name.rsplit('_',1)[0];fi=families.index(family);px=x+(.5+ks.index(K))*pw/len(ks)+(fi-(len(families)-1)/2)*.014;py=y+ph-ph*(mean-lo)/(hi-lo);color=colors[fi]
            f_line(slide,px,y+ph-ph*(mean-sd-lo)/(hi-lo),px,y+ph-ph*(mean+sd-lo)/(hi-lo),color)
            f_rect(slide,px-.025,py-.025,.05,.05,color)
        for j,K in enumerate(ks):f_text(slide,K,x+j*pw/len(ks),y+ph+.03,pw/len(ks),.18,7)
    f_text(slide,'All available atlas points: '+', '.join(families),8.6,7.3,7.2,.35,8)
    f_text(slide,'Points: participant mean ± SD; compact categorical parcel-count axis, no connecting lines. CoBRA450 / DRFC500 training cohorts disclosed.',.1,7.7,15.8,.2,8)
    out=w/'figures';out.mkdir(exist_ok=True);dest=out/'CoBRA_Editable_Comparisons.pptx';prs.save(dest)
    import zipfile
    with zipfile.ZipFile(dest) as archive:
        pictures=sum(archive.read(p).count(b'<p:pic>') for p in archive.namelist() if p.startswith('ppt/slides/slide') and p.endswith('.xml'))
        assert pictures==0
    f_save(out/'display_permutations.json',display);f_save(out/'editable_audit.json',dict(picture_objects=pictures,all_surface_polygons_native=True,statistics_native_shapes=True,subjects=ids))
    print(dest)
