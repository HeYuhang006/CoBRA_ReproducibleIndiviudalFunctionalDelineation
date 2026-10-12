function f_validate_initialization_guard_20261012(output)
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
code=fileparts(mfilename('fullpath'));layer=[code '/initialization_guard_20261012'];
addpath(genpath([root '/resources/rncut_author/source']));
addpath([root '/runs/rncut_native_interface_probe_20261008_v1/linux_mex']);
global RN_INIT_ORIGINAL_ALPHA RN_INIT_CACHE RN_INIT_AUDIT RN_INIT_SEED RN_INIT_ALPHA_CALLS;
RN_INIT_ORIGINAL_ALPHA=@alphaExpansion_v1;RN_INIT_SEED=20261009;
mkdir(output);cases={[4 4 4],3,19;[5 4 3],4,23};rows=cell(1,2);
for j=1:2
 if j>1,rmpath(layer);end
 s=cases{j,1};K=cases{j,2};seed=cases{j,3};mask=ones(s);n=numel(mask);
 rng(seed);X=randn(80,n);W=abs(corr(X));W(1:n+1:end)=0;
 rng(seed);native=RNcut(W,mask,K);
 RN_INIT_CACHE=[output '/continuous_case' num2str(j) '.mat'];
 RN_INIT_AUDIT=[output '/initialization_case' num2str(j)];RN_INIT_ALPHA_CALLS=0;
 addpath(layer,'-begin');rng(seed);guarded=RNcut(W,mask,K);
 assert(isequal(native,guarded),'Guard changed a valid native result');
 a=jsondecode(fileread([RN_INIT_AUDIT '.json']));assert(strcmp(a.selected,'original'));
 rng(seed);cached=RNcut(W,mask,K);
 assert(isequal(native,cached),'Cached continuous-eigenvector replay changed labels');
 rows{j}=struct('shape',s,'K',K,'seed',seed,'exact_output_labels',true,...
  'finite_alpha_calls',RN_INIT_ALPHA_CALLS,'initialization_unchanged',true,'cache_replay_exact',true);
 save([output '/native_case' num2str(j) '.mat'],'native','guarded','-v7');
end
badD=zeros(3,4);badD(3,:)=NaN;
badCases={{badD,zeros(3),zeros(1,3),sparse(4,4),ones(1,4)},...
 {zeros(3,4),zeros(3),[0 NaN 0],sparse(4,4),ones(1,4)},...
 {zeros(3,4),zeros(3),zeros(1,2),sparse(4,4),ones(1,4)}};
for j=1:numel(badCases)
 RN_INIT_AUDIT=[output '/negative_guard_case' num2str(j)];rejected=false;
 try,[~,~]=alphaExpansion_v1(badCases{j}{:});
 catch e,rejected=strcmp(e.identifier,'RNCut:InvalidOptimizerInput');end
 assert(rejected && isfile([RN_INIT_AUDIT '_invalid_optimizer_input.mat']),'Invalid input was not rejected and captured');
end
fid=fopen([output '/validation.json'],'w');fwrite(fid,jsonencode(struct('passed',true,'cases',{rows},'negative_tests',3,'invalid_inputs_rejected',true)));fclose(fid);
fprintf('INITIALIZATION_GUARD_TWO_NATIVE_CASES_EXACT\n');
end
