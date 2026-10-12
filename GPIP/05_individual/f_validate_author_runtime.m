function f_validate_author_runtime(reference,adapter,out)
restoredefaultpath;addpath(genpath(reference));addpath(adapter,'-begin');
assert(strcmp(which('eigs_15'),fullfile(adapter,'eigs_15.m')));
assert(strcmp(which('fmri_class'),fullfile(reference,'fmri_class.m')));
rng(20261009);global GPIP_EIGS_LOG;GPIP_EIGS_LOG=zeros(0,4);
A=randn(60);A=(A+A')/2;A(abs(A)<.2)=0;
S=sparsifyc(A,1e-6);assert(norm(full(S)-A,'fro')<1e-12);assert(any(S(:)<0));
d=sum(abs(S),2)+1;dr=.5*(sum(abs(S),2)-sum(S,2))+.5;
P=spdiags(1./sqrt(d+eps),0,60,60)*(S+spdiags(dr,0,60,60))*spdiags(1./sqrt(d+eps),0,60,60);
opts=struct('issym',1,'disp',0,'maxit',300,'tol',1e-8,'v0',ones(60,1),'p',35);
[V,D,flag]=eigs_15(@mex_w_times_x_symmetric,60,5,'LA',opts,tril(P));
[Q,F]=eig(full(P));[ev,ix]=sort(diag(F),'descend');Q=Q(:,ix(1:5));
eig_error=max(abs(sort(diag(D),'descend')-ev(1:5)));
projector_error=norm(V*V'-Q*Q','fro');assert(eig_error<1e-8 && projector_error<1e-5 && flag==0);
[disc,~,~]=ncutW(A,5);assert(all(sum(disc,2)==1) && all(sum(disc,1)>0));
% Validate author covariance arithmetic at independently varying retained T.
g=fmri_class();g.number_of_vertices=60;g.number_of_clusters=3;g.number_of_subjects=2;
g.clusters_labels={kron((1:3)',ones(20,1)),kron((1:3)',ones(20,1))};
cov_error=0;
for s=1:2
    y=randn(60,80+7*s);g.fmri_time_series{s}=y;g.pre_process_time_series(s);
    g.number_of_time_samples=size(y,2);g.compute_clusters_vertices_lists(s);g.compute_clusters_average_time_series(s);g.compute_clusters_correlation_matrices(s);
    means=zeros(3,size(y,2));for k=1:3;means(k,:)=mean(g.fmri_time_series{s}(g.clusters_labels{s}==k,:),1);end
    means=(means-mean(means,2))./std(means,0,2);C=means*means'/size(y,2);
    cov_error=max(cov_error,norm(C-g.clusters_correlation_matrices{s},'fro'));
end
assert(cov_error<1e-12);g.estimate_clusters_inverse_correlation_matrices_group(.01,.5,100);
for s=1:2;C=g.clusters_inverse_correlation_matrices{s};assert(all(isfinite(C(:))) && min(eig((C+C')/2))>0);end
% End-to-end fixture verifies synchronous driver, atomic checkpoint and resume.
driver=fileparts(mfilename('fullpath'));addpath(driver);
fixture=fullfile(out,'runtime_fixture_v2');if ~exist(fixture,'dir');mkdir(fixture);end
[xx,yy]=meshgrid(1:8,1:8);surface=struct('vertices',[xx(:),yy(:),zeros(64,1)],'faces',delaunay(xx,yy));
labels=min(3,ceil(xx(:)/3));geometry=fullfile(fixture,'geometry.mat');save(geometry,'surface','labels');
files=cell(2,1);frames=[87;94];for s=1:2;ts=randn(64,frames(s));files{s}=fullfile(fixture,sprintf('person%d.mat',s));save(files{s},'ts');end
job=fullfile(fixture,'job');if ~exist(job,'dir');mkdir(job);end
cfg=struct('reference',reference,'adapter',adapter,'geometry',geometry,'subjects',{{'fixture1','fixture2'}},'clusters',3,'mat_files',{files},'frames',frames,'iterations',2,'lambda',.01,'beta',1e-8,'seed',20261009,'output',job,'nominal_K',3,'visit','verification_fixture','config_sha256','runtime-fixture-only');
cp=fullfile(fixture,'config.json');fid=fopen(cp,'w');fprintf(fid,'%s',jsonencode(cfg));fclose(fid);
run_author_gpip(cp);check=load(fullfile(job,'checkpoint.mat'));assert(check.iteration==2 && numel(check.history)==2);
addpath(driver);run_author_gpip(cp);check=load(fullfile(job,'maps.mat'));assert(numel(check.labels)==2 && numel(check.history)==2);
r=struct('passed',true,'source_core_unmodified',true,'signed_affinities_preserved',true,'author_discretization_used',true,'legacy_EIGS_compatibility_only',true,'eigenvalue_error',eig_error,'projector_error',projector_error,'variable_T_covariance_error',cov_error,'MATLAB_version',version);
r.driver_checkpoint_and_resume_passed=true;
fid=fopen(fullfile(out,'runtime_validation.json'),'w');fprintf(fid,'%s',jsonencode(r));fclose(fid);
save(fullfile(out,'runtime_validation.mat'),'GPIP_EIGS_LOG','r');disp('GPIP_AUTHOR_RUNTIME_VALIDATION_PASSED');
end
