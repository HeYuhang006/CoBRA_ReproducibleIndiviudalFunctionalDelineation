function f_validate_rncut_single(output)
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
addpath(genpath([root '/resources/rncut_author/source']));
addpath([root '/runs/rncut_native_interface_probe_20261008_v1/linux_mex']);
rng(31);W=single(abs(corr(randn(80,64))));W(1:65:end)=0;
labels=RNcut(W,ones(4,4,4),3);assert(numel(unique(labels))==3);
fid=fopen(output,'w');fwrite(fid,jsonencode(struct('passed',true,'input_dtype','single','native_author_RNcut',true)));fclose(fid);
end
