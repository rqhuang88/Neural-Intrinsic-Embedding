function [S] = convert_to_mesh(X, F)
    S.nv = size(X,1);
    S.surface.TRIV = double(F);
    S.surface.X = X(:,1);
    S.surface.Y = X(:,2);
    S.surface.Z = X(:,3);
    S.surface.VERT = X;
    
end