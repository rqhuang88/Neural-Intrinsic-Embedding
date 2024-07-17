function [V,F] = load_raw_data(data_dir,data_name_list)
   num_of_data = size(data_name_list,1);
   V = {};
   F = {};
   for i =1:num_of_data
         [vertex,face] = read_off([data_dir,data_name_list(i,:)]);
         V{i} = vertex;
         F{i} = face;
   end
end