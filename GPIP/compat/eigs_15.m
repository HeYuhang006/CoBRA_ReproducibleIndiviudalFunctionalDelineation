function [V,D,flag] = eigs_15(afun,n,k,sigma,opts,varargin)
% MATLAB R2023a adapter for the distributed2013 EIGS copy.
% Same signed normalized-cut operator, eigenvalue selection and tolerances.
% No affinity clipping, objective change, label repair or altered discretizer.
if isa(afun,'function_handle')
    op = @(x) afun(x,varargin{:});
else
    op = afun;
end
[V,D,flag] = eigs(op,n,k,sigma,opts);
global GPIP_EIGS_LOG;
if isa(op,'function_handle')
    AV=zeros(size(V));
    for j=1:size(V,2);AV(:,j)=op(V(:,j));end
    residual = norm(AV-V*D,'fro')/max(1,norm(D,'fro'));
else
    residual = norm(op*V-V*D,'fro')/max(1,norm(D,'fro'));
end
GPIP_EIGS_LOG(end+1,:) = [n,k,flag,residual];
assert(flag==0 && residual<1e-6,'GPIP:EigensolverQC','Signed Ncut eigenproblem did not converge.');
end
