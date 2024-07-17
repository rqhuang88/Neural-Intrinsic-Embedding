function D = geodesics_to_all(shape, sources)
    S = shape.surface;
    
    sources = sources(:); 
    
    if(size(sources,2) > 1)
        error('sources must be stored in a vector');
    end
    
    D = comp_geodesics_to_all(double(S.X), double(S.Y), double(S.Z), ...
                             double(S.TRIV'), sources);
end