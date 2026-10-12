function f_test_rounding_diagnostic_20261012()
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
base=[root '/runs/group_nmf_rncut_20261009_v2/RNCut/k300_context_20261012'];
addpath(genpath([root '/resources/rncut_author/source']));
addpath([root '/runs/rncut_native_interface_probe_20261008_v1/linux_mex']);
rows=cell(1,22);
for j=1:22
 if j<=2
  s=load([root '/runs/group_nmf_rncut_20261009_v2/RNCut/case_' num2str(j) '.mat']);
  ks=[3 4];[input,~]=ncut(s.W,ks(j));
 else,rng(100+j);input=randn(100,10);end
 [native,nativeE]=discretisation(input);
 [replica,replicaE,tr]=f_discretisation_diagnostic_20261012(input,20,'first_vertex');
 assert(isequal(native,replica) && isequal(nativeE,replicaE),'Public-source replica mismatch');
 [long,~,longtr]=f_discretisation_diagnostic_20261012(input,200,'first_vertex');
 rng(20261009);[raw,~,rawtr]=f_discretisation_diagnostic_20261012(input,20,'author_raw');
 rows{j}=struct('case',j,'replica_exact',true,'K',size(input,2),...
  'native_occupied',nnz(sum(native,1)),'native_trace',tr,...
  'long_occupied',nnz(sum(long,1)),'long_trace',longtr,...
  'raw_occupied',nnz(sum(raw,1)),'raw_trace',rawtr);
end
fid=fopen([base '/rounding_diagnostic_tests.json'],'w');fwrite(fid,jsonencode(struct('cases',{rows},'diagnostic_only',true)));fclose(fid);
fprintf('PUBLIC_ROUNDING_REPLICA_22_CASES_EXACT\n');
end
