import os 
import hashlib
import shutil
import torch
import csv
import numpy as np
from PIL import Image
import pandas as pd

from sklearn.model_selection import train_test_split


PATH = './Data'

PATH_kinh_di = PATH + '/kinh_di'
PATH_tai_nan = PATH + '/tai_nan'

PATH_DATA = PATH + '/data'
os.makedirs(PATH_DATA, exist_ok=True)


# remove dulicate images 
hask_keys_kinh_di = {}
hask_keys_tai_nan = {}

for idx, img in enumerate(os.listdir(PATH_kinh_di)):
    img_path = os.path.join(PATH_kinh_di, img)
    if os.path.isfile(img_path):
        with open(img_path, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hask_keys_kinh_di:
            hask_keys_kinh_di[filehash] = idx 


for idx, img in enumerate(os.listdir(PATH_tai_nan)):
    img_path = os.path.join(PATH_tai_nan, img)
    if os.path.isfile(img_path):
        with open(img_path, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hask_keys_tai_nan:
            hask_keys_tai_nan[filehash] = idx 


# 0: kinh di, 1: tai nan

# load kinh_di images
idx_kinh_di = 0
idx_tai_nan = 0

for idx, img in enumerate(os.listdir(PATH_kinh_di)):
    # move and rename images
    if idx in hask_keys_kinh_di.values():
        original = os.path.join(PATH_kinh_di,img)
        extension = img.split('.')[-1]
        target = os.path.join(PATH_DATA,'kinh_di_{}.{}'.format(idx_kinh_di, extension))
        shutil.move(original, target)
        
        idx_kinh_di += 1


# load tai_nan images
for idx, img in enumerate(os.listdir(PATH_tai_nan)):
    # move and rename images
    if idx in hask_keys_tai_nan.values():
        original = os.path.join(PATH_tai_nan,img)
        extension = img.split('.')[-1]
        target = os.path.join(PATH_DATA,'tai_nan_{}.{}'.format(idx_tai_nan, extension))
        shutil.move(original, target)

        idx_tai_nan += 1
        if idx_tai_nan == 1996:
            break

print(idx_kinh_di)  # 1996 images
print(idx_tai_nan)  # 3992 images -> choose only 1996 images to balance


# make train, valid, test dataset

    
total_images = os.listdir(PATH_DATA)

train_images, valid_test_images = train_test_split(total_images, test_size= 0.6, random_state=42)
valid_images, test_images = train_test_split(valid_test_images, test_size= 0.5, random_state=42)

PATH_DATA_TRAIN = os.path.join(PATH_DATA, 'train')
PATH_DATA_TEST = os.path.join(PATH_DATA, 'test')
PATH_DATA_VALID = os.path.join(PATH_DATA, 'valid')


os.makedirs(os.path.join(PATH_DATA_TRAIN, 'kinh_di'), exist_ok=True)
os.makedirs(os.path.join(PATH_DATA_TRAIN, 'tai_nan'), exist_ok=True)
os.makedirs(os.path.join(PATH_DATA_TEST, 'kinh_di'), exist_ok=True)
os.makedirs(os.path.join(PATH_DATA_TEST, 'tai_nan'), exist_ok=True)
os.makedirs(os.path.join(PATH_DATA_VALID, 'kinh_di'), exist_ok=True)
os.makedirs(os.path.join(PATH_DATA_VALID, 'tai_nan'), exist_ok=True)




for img in train_images:
    if img.split('_')[0] == 'kinh':
        shutil.move(os.path.join(PATH_DATA, img), os.path.join(PATH_DATA_TRAIN, 'kinh_di'))
    else: 
        shutil.move(os.path.join(PATH_DATA, img), os.path.join(PATH_DATA_TRAIN, 'tai_nan'))
        
    

for img in test_images:
    if img.split('_')[0] == 'kinh':
        shutil.move(os.path.join(PATH_DATA, img), os.path.join(PATH_DATA_TEST, 'kinh_di'))
    else: 
        shutil.move(os.path.join(PATH_DATA, img), os.path.join(PATH_DATA_TEST, 'tai_nan'))



for img in valid_images:
    if img.split('_')[0] == 'kinh':
        shutil.move(os.path.join(PATH_DATA, img), os.path.join(PATH_DATA_VALID, 'kinh_di'))
    else: 
        shutil.move(os.path.join(PATH_DATA, img), os.path.join(PATH_DATA_VALID, 'tai_nan'))








