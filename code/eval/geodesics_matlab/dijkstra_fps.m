function idx = dijkstra_fps(shape, k)

if ~isfield(shape, 'surface')
    shape.surface.X = shape.X(:,1);
    shape.surface.Y = shape.X(:,2);
    shape.surface.Z = shape.X(:,3);
    shape.surface.TRIV = shape.T;
end

nv = size(shape.surface.X, 1);

idx = randi(nv,1);
dists = dijkstra_to_all(shape, idx);
idx = find(dists == max(dists), 1, 'first');

for i = 1:k-1
    dists = dijkstra_to_all(shape, idx);
    
    maxi = find(dists == max(dists), 1, 'first');
    idx = [idx; maxi];
end

idx = idx(1:end);