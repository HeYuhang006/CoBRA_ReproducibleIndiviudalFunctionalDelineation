function varargout=alphaExpansion_v1(Dcost,Scost,Lcost,Neighbor,iniLabel)
% Preserve the author solver. Stop before any NaN/Inf is silently cast to int32.
global RN_INIT_ORIGINAL_ALPHA RN_INIT_AUDIT RN_INIT_ALPHA_CALLS;
RN_INIT_ALPHA_CALLS=RN_INIT_ALPHA_CALLS+1;
K=size(Dcost,1);counts=histcounts(double(iniLabel(:)),.5:K+.5);
good=all(isfinite(Dcost(:))) && all(isfinite(Scost(:))) && ...
 all(isfinite(Lcost(:))) && all(isfinite(nonzeros(Neighbor))) && ...
 (numel(Lcost)==K || numel(Lcost)==1);
if ~good
 context=struct('Dcost',Dcost,'Scost',Scost,'Lcost',Lcost,'Neighbor',Neighbor,'iniLabel',iniLabel);
 save([RN_INIT_AUDIT '_invalid_optimizer_input.mat'],'context','counts','-v7');
 error('RNCut:InvalidOptimizerInput','Nonfinite or mismatched original optimizer inputs; saved and stopped without padding or label repair');
end
[varargout{1:nargout}]=feval(RN_INIT_ORIGINAL_ALPHA,Dcost,Scost,Lcost,Neighbor,iniLabel);
fprintf('RNCUT_FINITE_ALPHA_CALL %d K=%d\n',RN_INIT_ALPHA_CALLS,K);
end
