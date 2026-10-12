"""Package only complete measured comparisons, with file hashes for download."""
import csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace/runs/group_nmf_rncut_20261009_v2')


def f_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f_main():
    source=ROOT/'figure_data';complete=json.loads((source/'complete.json').read_text())
    assert complete==dict(state='complete',participants=50,maps=35,paired_tests=60)
    assert json.loads((ROOT/'RNCut/native_equivalence.json').read_text())['passed']
    assert json.loads((ROOT/'NMF/summary.json').read_text())['state']=='complete'
    assert json.loads((ROOT/'RNCut/summary.json').read_text())['state']=='complete'
    # Fixed group maps have one topology observation each, not 50 pseudo-replicates.
    maps=np.load(source/'atlas_labels_full_L.npz');summary=json.loads((source/'summary.json').read_text())
    edges=np.load(ROOT.parent/'full500_MSMAll_samplez/edges_L.npy');n=len(maps[maps.files[0]])
    graph=csr_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n));graph=graph+graph.T
    topology=[]
    for key,meta in summary['protocol']['metadata'].items():
        if meta['kind']!='multiscale':continue
        labels=maps[key];assigned=labels>=0;assert assigned.any()
        counts=[];largest=[]
        for label in np.unique(labels[assigned]):
            ids=np.flatnonzero(labels==label);number,components=connected_components(graph[ids][:,ids],directed=False)
            counts.append(number);largest.append(int(np.bincount(components).max()))
        topology.append(dict(map=key,K_nominal_bilateral=meta['K'],K_left_occupied=len(counts),assigned_vertices=int(assigned.sum()),
            disconnected_parcel_fraction=float(np.mean(np.asarray(counts)>1)),extra_components=int(np.sum(counts)-len(counts)),
            largest_component_vertex_fraction=float(np.sum(largest)/assigned.sum()),mean_components=float(np.mean(counts))))
    (source/'group_topology.json').write_text(json.dumps(dict(support='Full assigned left cortical support, before atlas-intersection masking',
        inference_unit='One fixed group atlas per map; descriptive, not subject-level replicates',maps=topology),indent=2))
    with (source/'group_topology.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(topology[0]));writer.writeheader();writer.writerows(topology)
    files={p.name:p for p in source.iterdir() if p.suffix in ['.json','.npy','.npz','.csv'] and p.name!='manifest.json'}
    files['native_equivalence.json']=ROOT/'RNCut/native_equivalence.json'
    manifest={name:dict(sha256=f_sha(path),bytes=path.stat().st_size) for name,path in files.items()}
    (source/'manifest.json').write_text(json.dumps(manifest,indent=2))
    bundle=ROOT/'group_figure_data.zip';temp=bundle.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in files.items():z.write(path,name)
        z.write(source/'manifest.json','manifest.json')
    temp.replace(bundle)
    print(json.dumps(dict(state='complete',path=str(bundle),sha256=f_sha(bundle),**{k:v for k,v in complete.items() if k!='state'})))


if __name__=='__main__':f_main()
