function f_test_empty_bound_20261012()
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
base=[root '/runs/group_nmf_rncut_20261009_v2/RNCut/k300_context_20261012'];
addpath(genpath([root '/resources/rncut_author/source']));
s=load([root '/runs/group_nmf_rncut_20261009_v1/interface_probe/KBound_1.mat']);
W=s.varargin{1};K=s.varargin{2};initial=double(s.varargin{3});flag=s.varargin{4};
ys={initial,min(initial,2),initial.*(initial~=2)+(initial==2)};
names={'full','missing_trailing3','missing_internal2'};rows=cell(1,3);
for j=1:3
 y=ys{j};d=KBound(W,K,y,flag);energy=NcutEnergy(W,y);
 rows{j}=struct('name',names{j},'occupied',unique(y)',...
  'cost_size',size(d),'nonfinite_per_column',sum(~isfinite(d),1),...
  'min_per_column',min(d,[],1),'max_per_column',max(d,[],1),...
  'energy',energy,'int32_nonfinite',double(int32(d(~isfinite(d))))');
 fprintf('%s nonfinite=%s NcutEnergy=%g\n',names{j},mat2str(sum(~isfinite(d),1)),energy);
end
s=load([base '/toy/labelcost_arguments.mat']);shape=s.inputs{1};opt=s.inputs{3};
rng(1234);checks=cell(1,100);
for j=1:100
 y=randi(opt.K,prod(shape),1);cost=calcLabelCost_v1(shape,y,opt);p=zeros(1,opt.K);
 for k=1:opt.K
  vol=false(shape);vol(opt.ind)=y==k;[~,p(k)]=bwlabeln(vol,26);
 end
 expected=p.*log(p);expected(p==0)=0;
 checks{j}=struct('components',p,'cost',cost,'expected',expected,...
  'max_abs_error',max(abs(cost-expected)));
end
save([base '/empty_bound_tests.mat'],'rows','checks','-v7');
fid=fopen([base '/empty_bound_tests.json'],'w');fwrite(fid,jsonencode(struct('rows',{rows},'cost_formula_checks',{checks})));fclose(fid);
fprintf('EMPTY_BOUND_AND_COMPONENT_FORMULA_TEST_FINISHED\n');
end
