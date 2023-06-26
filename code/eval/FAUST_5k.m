
load('models/FAUST_5k/FAUST_5k_test_on_faust_5k_sample_new');
set_basis = basis;
data_dir = 'data/FAUST_5k/off/';

data_name_list = ls ([data_dir,'*.off']);
data_name_list = data_name_list(81:end,:);
num_of_data = 20;
[V,Faces] = load_raw_data(data_dir,data_name_list);

M_name = 'models/FAUST_5k/M_FAUST_5k_test.mat';
vts_5k = {};
for i = 80:99
    vts = load(['data/FAUST_5k/corres/tr_reg_0',num2str(i,'%02d'),'.vts']);
    vts_5k{1,i-79} = vts;
end

%%
if exist(M_name)
    load(M_name);
else
    M = {}
    for i = 1:num_of_data
        S =convert_to_mesh(V{i}',Faces{i}');
        M{i} = geodesic_dense_matrx(S);
        count = i
    end
    save(M_name,'M');
end

%% full 
errors  = [];
errors_m = zeros(num_of_data);
errors_sub = [];
for src = 1:num_of_data
    for tar = 1:num_of_data
        if src == tar
            continue;
        end
        phiS= [squeeze(set_basis{src})];
        phiT =[squeeze(set_basis{tar})];      
    
        descS = [squeeze(desc{src})];
        descT = [squeeze(desc{tar})];
        
        G_desc_S =  phiS \ descS;
        G_desc_T =  phiT \ descT; 
        name1 = data_name_list(src,1:10);
        name2 = data_name_list(tar,1:10);
        C_ts = G_desc_S *  pinv(G_desc_T);
        C_st = G_desc_T *  pinv(G_desc_S);
        [idx,distance] = knnsearch(phiT,phiS*C_ts);  
        ind = sub2ind([size(phiT,1) size(phiT,1)],idx(vts_5k{src}),vts_5k{tar});
        M_T =  M{tar};
        geo_err = M_T(ind);
        geo_err = geo_err(:);  

        errors_m(src,tar) = mean(geo_err);
        errors  =  [errors ; geo_err];
    end
end

figure; imagesc( errors_m);
errors = errors(:);
figure; plot(sort(errors),linspace(0, 1, length(errors)) );

xlim([0, 0.25]);
legend(['OURS:', num2str(mean(errors),2)]);
set(gcf,'color','w');


