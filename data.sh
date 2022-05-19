find Data/TeamAnhTu_kinhdi -type f -name '*.jpg' -exec cp -r {} Data/kinh_di
find Data/TeamAnhTu_kinhdi -type f -name '*.jpg' -exec mv -t <new_folder> {} +
FOR /R "Data/TeamAnhTu_kinhdi" %i IN (*.jpg,*.png,*.jpeg,*.gif) DO MOVE "%i" "Data/kinh_di"