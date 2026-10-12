function f_rncut_surface_driver(mode,output)
% Unchanged author RNcut with two explicit graph-domain spatial interfaces.
root='/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace';
code=fileparts(mfilename('fullpath'));
addpath(genpath([root '/resources/rncut_author/source']));
addpath([root '/runs/rncut_native_interface_probe_20261008_v1/linux_mex']);
global RN_SURFACE_ACTIVE RN_AUTHOR_NEIGHBORS RN_AUTHOR_BWLABELN;
RN_SURFACE_ACTIVE=false;
RN_AUTHOR_NEIGHBORS=str2func('getNeighbours');RN_AUTHOR_BWLABELN=str2func('bwlabeln');
mkdir(output);warning('off','all');
if strcmp(mode,'validate')
    cases={[4 4 4],3,19;[5 4 3],4,23};records=cell(1,size(cases,1));
    for ci=1:size(cases,1)
        RN_SURFACE_ACTIVE=false;
        shape=cases{ci,1};K=cases{ci,2};seed=cases{ci,3};mask=ones(shape);n=numel(mask);
        rng(seed);X=randn(80,n);W=abs(corr(X));W(1:n+1:end)=0;
        rng(seed);native=RNcut(W,mask,K);
        neighbors=feval(RN_AUTHOR_NEIGHBORS,shape,find(mask));
        f_activate(mask,neighbors);
        addpath([code '/surface_adapter'],'-begin');
        % Component costs must agree, including scattered/empty components.
        ccerrors=0;
        for test=1:30
            B=rand(shape)>.65;
            [a,na]=feval(RN_AUTHOR_BWLABELN,B,26);[b,nb]=bwlabeln(B,26);
            partitionErrors=nnz((a(:)==a(:)')~=(b(:)==b(:)'));
            if partitionErrors>0 || na~=nb
                save([output '/component_mismatch_' num2str(ci) '_' num2str(test) '.mat'],'a','b','na','nb','B','neighbors','-v7');
            end
            ccerrors=ccerrors+partitionErrors+abs(double(na)-double(nb));
        end
        assert(ccerrors==0,'Graph components differ from native 26-connectivity');
        rng(seed);adapted=RNcut(W,mask,K);
        assert(isequal(native,adapted),'Graph-adapter RNcut differs from native voxel labels');
        records{ci}=struct('shape',shape,'K',K,'seed',seed,'component_tests',30,...
            'component_errors',ccerrors,'label_errors',nnz(native~=adapted));
        save([output '/case_' num2str(ci) '.mat'],'native','adapted','W','mask','-v7');
        fprintf('PASS NATIVE EQUIVALENCE case=%d K=%d\n',ci,K);
    end
    f_json([output '/native_equivalence.json'],struct('passed',true,'cases',{records},...
        'unchanged_author_functions',{{'RNcut','ncut','ncutW','KBound','NcutEnergy','getPairsCost','calcLabelCost_v1','alphaExpansion_v1'}},...
        'adapted_interfaces',{{'getNeighbours','bwlabeln'}},...
        'scope','Exact native voxel diagnostics. Real-data mesh topology replaces voxel 26-neighborhood; not the original volumetric experiment.'));
    return;
end
assert(strcmp(mode,'fit'));
verification=jsondecode(fileread([output '/native_equivalence.json']));assert(verification.passed);
cfg=jsondecode(fileread([output '/input.json']));n=cfg.vertices;
fid=fopen(cfg.affinity_npy,'rb');assert(fid>=0);fseek(fid,cfg.header_bytes,'bof');
W=fread(fid,[n n],'single=>single');fclose(fid);assert(isequal(size(W),[n n]));
assert(all(isfinite(W(:))) && min(W(:))>=0);
assert(max(max(abs(W(1:151:end,:)-W(:,1:151:end)')))<1e-6,'Affinity is not symmetric');
assert(all(diag(W)==0));
edges=load(cfg.edges_mat);edges=double(edges.edges)+1;
adj=sparse([edges(:,1);edges(:,2)],[edges(:,2);edges(:,1)],1,n,n);adj=spones(adj);
neighbors=cell(n,1);
for i=1:n,neighbors{i}=find(adj(i,:));end
% The volume is only an index container. Neither regularizer uses its geometry.
mask=zeros(ceil(n/4),2,2);mask(1:n)=1;
f_activate(mask,neighbors);addpath([code '/surface_adapter'],'-begin');
mkdir([output '/maps']);
for k=[50 100 150 200]
    dest=[output '/maps/labels_L_' num2str(k) '.mat'];
    if isfile(dest),continue;end
    f_json([output '/status.json'],struct('state','running','phase','author_RNcut_optimization','K_nominal_bilateral',2*k,'vertices',n));
    rng(cfg.seed);started=tic;labelsVolume=RNcut(W,mask,k);seconds=toc(started);
    labels=double(labelsVolume(1:n));labels=labels(:)-1;
    occupied=numel(unique(labels));
    % Preserve original output: no CoBRA connected repair, splitting, or reseeding.
    save(dest,'labels','seconds','occupied','-v7');
    f_json([output '/maps/labels_L_' num2str(k) '.json'],struct('K_requested_left',k,...
        'K_occupied_left',occupied,'seconds',seconds,'native_optimizer_unchanged',true,...
        'surface_domain_adaptation',true,'no_label_repair',true));
    fprintf('RNcut COMPLETE K=%d occupied=%d seconds=%.3f\n',2*k,occupied,seconds);
end
f_json([output '/optimizer_complete.json'],struct('state','complete','native_voxel_equivalence_passed',true,'vertices',n));
end

function f_activate(mask,neighbors)
global RN_SURFACE_ACTIVE RN_SURFACE_NEIGHBORS RN_SURFACE_GRAPH RN_SURFACE_IND RN_SURFACE_CONTAINER_N;
RN_SURFACE_IND=find(mask);n=numel(RN_SURFACE_IND);assert(numel(neighbors)==n);
RN_SURFACE_NEIGHBORS=neighbors;
counts=cellfun(@numel,neighbors);ii=repelem((1:n)',counts);
columns=cellfun(@(x) double(x(:)),neighbors,'UniformOutput',false);jj=vertcat(columns{:});
RN_SURFACE_GRAPH=graph(sparse(ii,jj,1,n,n),'upper');
RN_SURFACE_CONTAINER_N=numel(mask);RN_SURFACE_ACTIVE=true;
end

function f_json(path,data)
fid=fopen([path '.tmp'],'w');assert(fid>=0);fwrite(fid,jsonencode(data));fclose(fid);movefile([path '.tmp'],path,'f');
end
