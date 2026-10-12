function [Y,E,values]=ncutW(W,K)
% Original continuous Ncut and rounding, with a declared initialization fallback.
% No output labels are split, merged, reassigned, or repaired.
global RN_INIT_CACHE RN_INIT_AUDIT RN_INIT_SEED;
assert(~isempty(RN_INIT_CACHE) && ~isempty(RN_INIT_AUDIT));
matrix_sha256=f_matrix_sha256(W);
if isfile(RN_INIT_CACHE)
 s=load(RN_INIT_CACHE);assert(s.K==K && isequal(s.matrix_size,size(W)) && ...
  isfield(s,'matrix_sha256') && strcmp(s.matrix_sha256,matrix_sha256),'Cached matrix identity mismatch');
 continuous=s.continuous;values=s.values;rng(s.after_ncut_rng);
else
 [continuous,values]=ncut(W,K);after_ncut_rng=rng;matrix_size=size(W);
 save(RN_INIT_CACHE,'continuous','values','K','matrix_size','matrix_sha256','after_ncut_rng','-v7');
end
[original,originalE]=discretisation(continuous);
[replica,replicaE,originalTrace]=f_discretisation_diagnostic_20261012(continuous,20,'first_vertex');
assert(isequal(original,replica) && isequal(originalE,replicaE),'Public rounding replica must agree exactly');
Y=original;E=originalE;selected='original';extendedTrace=[];rawTrace=[];
if nnz(sum(Y,1))~=K
 % First try additional iterations of the exact same deterministic solver.
 [candidate,candidateE,extendedTrace]=f_discretisation_diagnostic_20261012(continuous,200,'first_vertex');
 if nnz(sum(candidate,1))==K
  Y=candidate;E=candidateE;selected='same_solver_extended_cap200';
 else
  % One predeclared fallback: the author's supplied commented 'raw code'
  % first-center rule. Same fixed seed, no seed search or evaluation tuning.
  current_rng=rng;rng(RN_INIT_SEED);
  [candidate,candidateE,rawTrace]=f_discretisation_diagnostic_20261012(continuous,20,'author_raw');
  rng(current_rng);
  Y=candidate;E=candidateE;selected='author_raw_first_center_fixed_seed';
 end
end
counts=full(sum(Y,1));
audit=struct('K_requested',K,'K_occupied_original',nnz(sum(original,1)),...
 'K_occupied_selected',nnz(counts),'selected',selected,'counts',counts,...
 'original_trace',originalTrace,'extended_trace',extendedTrace,'raw_trace',rawTrace,...
 'no_posthoc_label_repair',true,'seed_search',false,'seed',RN_INIT_SEED,'matrix_sha256',matrix_sha256);
save([RN_INIT_AUDIT '.mat'],'original','Y','E','values','audit','-v7');
fid=fopen([RN_INIT_AUDIT '.json'],'w');fwrite(fid,jsonencode(audit));fclose(fid);
assert(nnz(counts)==K,'Both predeclared author-rounding fallbacks still have an unused label; stop');
Y=full(Y);
fprintf('RNCUT_INITIALIZATION requested=%d original=%d selected=%d variant=%s\n',K,nnz(sum(original,1)),nnz(counts),selected);
end
function digest=f_matrix_sha256(W)
% Full numeric matrix identity, streamed to avoid a second full-matrix copy.
md=java.security.MessageDigest.getInstance('SHA-256');
md.update(int8([class(W) ':' sprintf('%d,',size(W))]));
for first=1:2^22:numel(W)
 block=W(first:min(numel(W),first+2^22-1));
 md.update(typecast(block(:),'int8'));
end
digest=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
