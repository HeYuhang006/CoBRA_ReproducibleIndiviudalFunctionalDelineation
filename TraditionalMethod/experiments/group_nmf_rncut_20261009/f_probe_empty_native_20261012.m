function f_probe_empty_native_20261012()
% Deliberately empty one INITIAL cluster in a tiny synthetic fixture only.
% Observe how original RNcut handles it; no production labels are changed.
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
base=[root '/runs/group_nmf_rncut_20261009_v2/RNCut/k300_context_20261012'];
addpath(genpath([root '/resources/rncut_author/source']));
addpath([root '/runs/rncut_native_interface_probe_20261008_v1/linux_mex']);
global RN_CTX_ORIGINAL_GCO RN_CTX_ORIGINAL_COST RN_CTX_OUTPUT RN_EMPTY_ORIGINAL_NCUT;
RN_CTX_ORIGINAL_GCO=@GCO_SetLabelCost;RN_CTX_ORIGINAL_COST=@calcLabelCost_v1;
RN_EMPTY_ORIGINAL_NCUT=@ncutW;
RN_CTX_OUTPUT=[base '/toy_empty'];mkdir(RN_CTX_OUTPUT);
adapter=[RN_CTX_OUTPUT '/observer'];mkdir(adapter);
fid=fopen([adapter '/ncutW.m'],'w');
fprintf(fid,['function varargout=ncutW(varargin)\n'...
 'global RN_EMPTY_ORIGINAL_NCUT;\n'...
 '[varargout{1:nargout}]=feval(RN_EMPTY_ORIGINAL_NCUT,varargin{:});\n'...
 'y=varargout{1};y(:,2)=y(:,2)+y(:,3);y(:,3)=0;varargout{1}=y;\n'...
 'end\n']);fclose(fid);
addpath([base '/observer'],'-begin');addpath(adapter,'-begin');
rng(19);mask=ones(4,4,4);n=numel(mask);X=randn(80,n);W=abs(corr(X));W(1:n+1:end)=0;
rng(19);
try
 RNcut(W,mask,3);error('Test must exercise the original empty-cluster failure');
catch e
 assert(contains(e.message,'LabelCost must be scalar or of length NumLabels'),e.message);
 fprintf('SYNTHETIC_EMPTY_NATIVE_FAILURE_CAPTURED\n');
end
end
