function neighbors=getNeighbours(dim,ind)
% Replace only the spatial domain, preserving the author's affinity/optimizer.
global RN_SURFACE_ACTIVE RN_SURFACE_NEIGHBORS RN_SURFACE_IND RN_AUTHOR_NEIGHBORS;
if ~isempty(RN_SURFACE_ACTIVE) && RN_SURFACE_ACTIVE
    assert(isequal(double(ind(:)),double(RN_SURFACE_IND(:))),'Surface container ordering changed');
    neighbors=RN_SURFACE_NEIGHBORS;
else
    neighbors=feval(RN_AUTHOR_NEIGHBORS,dim,ind);
end
end
