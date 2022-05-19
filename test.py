import glob
import os

count = 0
for filename in glob.iglob('./Data/TeamAnhTu_kinhdi/' + '**/*.jpg', recursive=True):
    count += 1
    # print(filename)

for filename in glob.iglob('./Data/TeamAnhTu_kinhdi/' + '**/*.png', recursive=True):
    count += 1
    # print(os.path.basename(filename))

for filename in glob.iglob('./Data/TeamAnhTu_kinhdi/' + '**/*.jpeg', recursive=True):
    count += 1
    
for filename in glob.iglob('./Data/TeamAnhTu_kinhdi/' + '**/*.gif', recursive=True):
    count += 1
# list = glob.glob('./Data/TeamAnhTu_kinhdi/' + '**/*.jpg', recursive=True)
# print(len(list))

print(count)

# import pathlib
# import shutil

# def move_files(source_folder:pathlib.Path, target_folder:pathlib.Path):
#     target_folder.mkdir(parents=True, exist_ok=True)
#     for image_file in source_folder.rglob("*.jpg"): # recursively find image paths
#         shutil.copy(image_file, target_folder.joinpath(image_file.name))

# move_files(pathlib.Path('Data/TeamAnhTu_kinhdi'), pathlib.Path('Data/kinh_di'))