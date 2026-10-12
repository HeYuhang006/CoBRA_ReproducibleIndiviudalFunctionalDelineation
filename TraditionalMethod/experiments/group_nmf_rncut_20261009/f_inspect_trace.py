"""Inspect public author-code diagnostic inputs and outputs, never decrypt P-code."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy import sparse
root=Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace/runs/group_nmf_rncut_20261009_v1/interface_probe')
for p in sorted(root.glob('*.mat')):
    if p.name=='native_trace_result.mat':continue
    print('\nFILE',p.name)
    a=loadmat(p)
    for key in ['varargin','varargout']:
        for index,x in enumerate(a[key].ravel()):
            if sparse.issparse(x):print(key,index,'sparse',x.shape,'nnz',x.nnz,'data',x.data[:15])
            elif x.size<100:print(key,index,x.dtype,x.shape,x)
            else:print(key,index,x.dtype,x.shape,'range',x.min(),x.max(),'first',x.ravel(order='F')[:12])
result=loadmat(root/'native_trace_result.mat',simplify_cells=True)
print('CALLS',result['RN_TRACE_CALLS'])
print('PROFILE',[f['FunctionName'] for f in result['prof']['FunctionTable'] if not f['FunctionName'].startswith(('GCO','gco','eigs'))])
