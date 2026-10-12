function [L,count]=bwlabeln(BW,varargin)
% Connected components on the declared spatial graph, rather than a fake grid.
global RN_SURFACE_ACTIVE RN_SURFACE_GRAPH RN_SURFACE_IND RN_SURFACE_CONTAINER_N RN_AUTHOR_BWLABELN;
if isempty(RN_SURFACE_ACTIVE) || ~RN_SURFACE_ACTIVE
    [L,count]=feval(RN_AUTHOR_BWLABELN,BW,varargin{:});return;
end
assert(numel(BW)==RN_SURFACE_CONTAINER_N,'Unexpected author spatial component input');
assert(numel(varargin)==1 && isequal(double(varargin{1}),26),'Unexpected author connectivity convention');
L=zeros(size(BW));selected=find(BW(RN_SURFACE_IND));
if isempty(selected),count=0;return;end
sub=subgraph(RN_SURFACE_GRAPH,selected);
components=conncomp(sub);count=max(components);
% Component IDs are arbitrary; number mesh components by their first vertex.
% The author label-cost function uses their count, not the component ID order.
first=accumarray(components(:),double(selected(:)),[count 1],@min);
[~,order]=sort(first);renumber=zeros(count,1);renumber(order)=1:count;
components=renumber(components);
L(RN_SURFACE_IND(selected))=components;
end
