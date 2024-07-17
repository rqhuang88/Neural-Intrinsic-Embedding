function M = geodesic_dense_matrx(S)

nv = length(S.surface.X);
D = cell(1, nv); 
parfor i = 1:nv
   	D{i} = geodesics_to_all(S, i); 
end

M = cell2mat(D); 