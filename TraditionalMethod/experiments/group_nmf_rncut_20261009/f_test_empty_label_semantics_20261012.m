function f_test_empty_label_semantics_20261012()
% Black-box tests of the unchanged author's public cost function.
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
base=[root '/runs/group_nmf_rncut_20261009_v2/RNCut/k300_context_20261012'];
addpath(genpath([root '/resources/rncut_author/source']));
s=load([base '/toy/labelcost_arguments.mat']);
shape=s.inputs{1}; original=s.inputs{2}; options=s.inputs{3};
labels={original, min(original,2), original.*(original~=2)+(original==2),...
    ones(size(original)), ones(size(original))*2, ones(size(original))*3};
names={'full','missing_trailing3','missing_internal2','only1','only2','only3'};
rows=cell(0);record=0;
for k=[3 4 5]
    opt=options;opt.K=k;
    for j=1:numel(labels)
        y=labels{j};record=record+1;
        row=struct('name',names{j},'options_K',k,'occupied',unique(y(:))',...
            'output',[],'error','');
        try
            cost=calcLabelCost_v1(shape,y,opt);
            row.output=cost;row.output_size=size(cost);
        catch e
            row.error=e.message;
        end
        rows{record}=row;
        fprintf('%s optionsK=%d cost=%s error=%s\n',row.name,k,mat2str(row.output),row.error);
    end
end
% A missing internal ID and a missing trailing ID differ only by ID permutation.
y=min(original,2);swap=[1 3 2];yp=reshape(swap(y(:)),size(y));
c=calcLabelCost_v1(shape,y,options);cp=calcLabelCost_v1(shape,yp,options);
permutation=struct('labels12_cost',c,'labels13_cost',cp,'swap',swap);
save([base '/empty_label_semantics.mat'],'rows','permutation','options','shape','original','-v7');
f_json([base '/empty_label_semantics.json'],struct('cases',{rows},'permutation',permutation));
fprintf('AUTHOR_EMPTY_LABEL_SEMANTICS_TEST_FINISHED\n');
end
function f_json(path,data)
fid=fopen(path,'w');assert(fid>=0);fwrite(fid,jsonencode(data));fclose(fid);
end
