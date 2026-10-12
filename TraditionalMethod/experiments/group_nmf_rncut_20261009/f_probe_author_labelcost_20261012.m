function f_probe_author_labelcost_20261012()
% Observe the original function's actual inputs on the existing native fixture.
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
base=[root '/runs/group_nmf_rncut_20261009_v2/RNCut/k300_context_20261012'];
addpath(genpath([root '/resources/rncut_author/source']));
addpath([root '/runs/rncut_native_interface_probe_20261008_v1/linux_mex']);
global RN_CTX_ORIGINAL_GCO RN_CTX_ORIGINAL_COST RN_CTX_OUTPUT;
RN_CTX_ORIGINAL_GCO=@GCO_SetLabelCost;RN_CTX_ORIGINAL_COST=@calcLabelCost_v1;
RN_CTX_OUTPUT=[base '/toy'];mkdir(RN_CTX_OUTPUT);
addpath([base '/observer'],'-begin');
rng(19);mask=ones(4,4,4);n=numel(mask);X=randn(80,n);W=abs(corr(X));W(1:n+1:end)=0;
rng(19);native=RNcut(W,mask,3);
save([RN_CTX_OUTPUT '/native_result.mat'],'native','mask','-v7');
fprintf('TOY_AUTHOR_LABELCOST_CAPTURE_OK\n');
end
