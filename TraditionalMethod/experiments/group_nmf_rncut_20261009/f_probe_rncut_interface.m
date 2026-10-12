function f_probe_rncut_interface(output)
% Read-only interface trace of the unchanged author RNcut on its voxel domain.
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
addpath(genpath([root '/resources/rncut_author/source']));
addpath([root '/runs/rncut_native_interface_probe_20261008_v1/linux_mex']);
global RN_TRACE_ORIGINAL RN_TRACE_CALLS RN_TRACE_OUTPUT;
RN_TRACE_OUTPUT=output; mkdir(output); RN_TRACE_CALLS=struct(); RN_TRACE_ORIGINAL=struct();
names={'getNeighbours','getSub','getPairsCost','getPairsCost_FCS','adjustNeighborWeight','KBound','NcutEnergy','calcLabelCost_v1','alphaExpansion_v1','bwlabeln'};
wrap=[output '/wrappers'];mkdir(wrap);
for i=1:numel(names)
    name=names{i}; RN_TRACE_ORIGINAL.(name)=str2func(name);
    info=functions(RN_TRACE_ORIGINAL.(name));fprintf('ORIGINAL %s %s\n',name,info.file);
    text=sprintf(['function varargout=%s(varargin)\n' ...
      'global RN_TRACE_ORIGINAL RN_TRACE_CALLS RN_TRACE_OUTPUT;\n' ...
      'if ~isfield(RN_TRACE_CALLS,''%s''),RN_TRACE_CALLS.%s=0;end\n' ...
      'RN_TRACE_CALLS.%s=RN_TRACE_CALLS.%s+1;\n' ...
      '[varargout{1:nargout}]=feval(RN_TRACE_ORIGINAL.%s,varargin{:});\n' ...
      'if RN_TRACE_CALLS.%s<=2,save([RN_TRACE_OUTPUT ''/%s_'' num2str(RN_TRACE_CALLS.%s) ''.mat''],''varargin'',''varargout'',''-v7'');end\n' ...
      'end\n'],name,name,name,name,name,name,name,name,name);
    fid=fopen([wrap '/' name '.m'],'w');fwrite(fid,text);fclose(fid);
end
addpath(wrap,'-begin');
rng(19);X=randn(80,64);W=abs(corr(X));W(1:65:end)=0;mask=ones(4,4,4);
warning('off','all'); profile on;
labels=RNcut(W,mask,3);prof=profile('info');profile off;
save([output '/native_trace_result.mat'],'labels','W','mask','RN_TRACE_CALLS','prof','-v7');
fid=fopen([output '/trace_complete.json'],'w');fwrite(fid,jsonencode(struct('state','complete','calls',RN_TRACE_CALLS)));fclose(fid);
fprintf('TRACE COMPLETE\n');
end
