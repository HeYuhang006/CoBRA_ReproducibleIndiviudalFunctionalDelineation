function f_run_author_gpip(config_path)
cfg=jsondecode(fileread(config_path));restoredefaultpath;addpath(genpath(cfg.reference));addpath(cfg.adapter,'-begin');
assert(strcmp(which('fmri_class'),fullfile(cfg.reference,'fmri_class.m')));
assert(strcmp(which('ncutW'),fullfile(cfg.reference,'m_lib','Ncut_9','ncutW.m')));
assert(strcmp(which('eigs_15'),fullfile(cfg.adapter,'eigs_15.m')));
geom=load(cfg.geometry);g=fmri_class();g.set_cortical_surface(geom.surface);
n=numel(cfg.subjects);K=cfg.clusters;initial=double(geom.labels(:));
for s=1:n;g.set_clusters_labels(initial,s,K);end
g.fmri_time_series=cell(n,1);history=struct([]);iteration=0;
checkpoint=fullfile(cfg.output,'checkpoint.mat');
if exist(checkpoint,'file')
    previous=load(checkpoint);assert(strcmp(previous.config_sha256,cfg.config_sha256));
    g.clusters_labels=previous.labels;iteration=previous.iteration;history=previous.history;
end
global GPIP_EIGS_LOG;GPIP_EIGS_LOG=zeros(0,4);
for it=iteration+1:cfg.iterations
    tic;old=g.clusters_labels;
    g.boundary_vertices=cell(n,1);g.boundary_faces=cell(n,1);g.boundary_rois=cell(n,1);
    for s=1:n
        report(cfg,'parcel_covariance',it,s);
        a=load(cfg.mat_files{s},'ts');g.fmri_time_series{s}=double(a.ts);g.pre_process_time_series(s);
        % Author class stores a scalar T. Censoring creates varying retained T:
        % set it to this subject's actual T before its unchanged methods.
        g.number_of_time_samples=size(a.ts,2);assert(g.number_of_time_samples==cfg.frames(s));
        g.compute_clusters_vertices_lists(s);g.compute_clusters_average_time_series(s);g.compute_clusters_correlation_matrices(s);
        g.fmri_time_series{s}=[];clear a;
    end
    report(cfg,'group_precision',it,0);g.estimate_clusters_inverse_correlation_matrices_group(cfg.lambda,.5,100);
    next=cell(n,1);changed=zeros(n,1);occupied=zeros(n,1);boundary=zeros(n,1);
    for s=1:n
        report(cfg,'signed_Ncut',it,s);
        a=load(cfg.mat_files{s},'ts');g.fmri_time_series{s}=double(a.ts);g.pre_process_time_series(s);g.number_of_time_samples=size(a.ts,2);
        g.find_boundary_vertices(s);instance=g.create_subject_instance(s);boundary(s)=numel(instance.boundary_vertices);
        rng(cfg.seed+it*1000+s,'twister');
        next{s}=ooc_estimate_parcellation(instance,cfg.beta);
        assert(all(next{s}>=0 & next{s}<=K));assert(all(next{s}(initial==0)==0));
        changed(s)=mean(next{s}(initial>0)~=old{s}(initial>0));occupied(s)=numel(unique(next{s}(next{s}>0)));
        g.fmri_time_series{s}=[];clear a instance;
    end
    % Synchronous/Jacobi update exactly as the author's outer loop: all
    % subject refinements used the previous iteration's group precisions/maps.
    g.clusters_labels=next;labels=next;iteration=it;config_sha256=cfg.config_sha256;
    h=struct('iteration',it,'changed_fraction',changed,'occupied_clusters',occupied,'boundary_vertices',boundary,'elapsed_seconds',toc);
    if isempty(history);history=h;else;history(it)=h;end
    save(fullfile(cfg.output,sprintf('iteration_%02d.mat',it)),'labels','iteration','h','GPIP_EIGS_LOG','-v7');
    temp=fullfile(cfg.output,'checkpoint.tmp.mat');save(temp,'labels','iteration','history','config_sha256','-v7');movefile(temp,checkpoint,'f');
    report(cfg,'iteration_complete',it,n);disp(jsonencode(h));
end
labels=g.clusters_labels;save(fullfile(cfg.output,'maps.mat'),'labels','history','GPIP_EIGS_LOG','-v7');
r=struct('state','complete','subjects',n,'iterations',cfg.iterations,'clusters',K,'config_sha256',cfg.config_sha256,'source_core_unmodified',true,'independent_visit',true,'no_custom_connectivity_guard',true,'no_empty_cluster_repair',true,'runtime_compatibility_disclosed',true);
fid=fopen(fullfile(cfg.output,'complete.json'),'w');fprintf(fid,'%s',jsonencode(r));fclose(fid);report(cfg,'complete',cfg.iterations,n);
end

function report(cfg,phase,it,s)
r=struct('state','running','phase',phase,'iteration',it,'subject_index',s,'subjects',numel(cfg.subjects),'K',cfg.nominal_K,'visit',cfg.visit,'updated_unix',posixtime(datetime('now','TimeZone','UTC')));
if strcmp(phase,'complete');r.state='complete';end
temp=fullfile(cfg.output,'status.tmp.json');fid=fopen(temp,'w');fprintf(fid,'%s',jsonencode(r));fclose(fid);movefile(temp,fullfile(cfg.output,'status.json'),'f');
end
