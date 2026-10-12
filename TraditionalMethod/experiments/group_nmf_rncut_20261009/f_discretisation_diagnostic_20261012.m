function [Y,E,trace]=f_discretisation_diagnostic_20261012(E,maxIterations,centerMode)
% Diagnostic copy of the PUBLIC author discretisation.m (not P-code).
% All algebra is unchanged. Two explicitly selected diagnostic variants:
% 1. increase only the original iteration cap;
% 2. use the author's commented 'raw code' first-center rule, with caller RNG.
% Never selects variants using evaluation data; never repairs output labels.
[n,k]=size(E);vm=sqrt(sum(E.*E,2));E=E./repmat(vm,1,k);
R=zeros(k);
if strcmp(centerMode,'first_vertex'),first=1;
elseif strcmp(centerMode,'author_raw'),first=1+round(rand(1)*(n-1));
else,error('Unknown diagnostic centerMode');end
R(:,1)=E(first,:)';c=zeros(n,1);
for j=2:k
 c=c+abs(E*R(:,j-1));[~,i]=min(c);R(:,j)=E(i,:)';
end
last=0;trace=struct('first_vertex',first,'center_mode',centerMode,...
 'iteration',[],'objective',[],'occupied',[],'reason','');
for iteration=1:maxIterations+1
 Y=discretisationEigenVectorData(E*R);
 [U,S,V]=svd(Y'*E,0);objective=2*(n-trace_builtin(S));
 trace.iteration(end+1)=iteration;trace.objective(end+1)=objective;
 trace.occupied(end+1)=nnz(sum(Y,1));
 if abs(objective-last)<eps
  trace.reason='original_objective_tolerance';break;
 elseif iteration>maxIterations
  trace.reason='iteration_cap';break;
 else,last=objective;R=V*U';end
end
end
function s=trace_builtin(x)
s=sum(diag(x));
end
