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
PATH_LABEL = PATH + '/label'
os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_LABEL, exist_ok=True)
# PATH_LABEL = os.path.join(PATH_LABEL, 'labels.csv')


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




with open(os.path.join(PATH_LABEL, 'labels.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    # 0: kinh di, 1: tai nan

    # load kinh_di images
    idx_kinh_di = 0
    idx_tai_nan = 0

    for idx, img in enumerate(os.listdir(PATH_kinh_di)):
        # move and rename images
        if idx in hask_keys_kinh_di.values():
            original = os.path.join(PATH_kinh_di,img)
            target = os.path.join(PATH_DATA,f'kinh_di_{idx_kinh_di}.jpg')
            shutil.move(original, target)
            
            writer.writerow([f'kinh_di_{idx_kinh_di}.jpg', 0])
            idx_kinh_di += 1


    # load tai_nan images
    for idx, img in enumerate(os.listdir(PATH_tai_nan)):
        # move and rename images
        if idx in hask_keys_tai_nan.values():
            original = os.path.join(PATH_tai_nan,img)
            target = os.path.join(PATH_DATA,f'tai_nan_{idx_tai_nan}.jpg')
            shutil.move(original, target)

            writer.writerow([f'tai_nan_{idx_tai_nan}.jpg', 1])
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

PATH_LABEL_TRAIN = os.path.join(PATH_LABEL, 'train')
PATH_LABEL_TEST = os.path.join(PATH_LABEL, 'test')
PATH_LABEL_VALID = os.path.join(PATH_LABEL, 'valid')

os.makedirs(PATH_DATA_TRAIN, exist_ok=True)
os.makedirs(PATH_DATA_TEST, exist_ok=True)
os.makedirs(PATH_DATA_VALID, exist_ok=True)

os.makedirs(PATH_LABEL_TRAIN, exist_ok=True)
os.makedirs(PATH_LABEL_TEST, exist_ok=True)
os.makedirs(PATH_LABEL_VALID, exist_ok=True)

labels = pd.read_csv(os.path.join(PATH_LABEL, 'labels.csv'), names=['name', 'label'], index_col=0)
with open(os.path.join(PATH_LABEL_TRAIN, 'labels_train.csv'), 'w', newline='') as f:
    writer = csv.writer(f) 
    for img in train_images:
        
        shutil.move(os.path.join(PATH_DATA, img), PATH_DATA_TRAIN)
        
        writer.writerow([img, labels.loc[img].label])

with open(os.path.join(PATH_LABEL_TEST, 'labels_test.csv'), 'w', newline='') as f:
    writer = csv.writer(f) 
    for img in test_images:
        shutil.move(os.path.join(PATH_DATA, img), PATH_DATA_TEST)

        writer.writerow([img, labels.loc[img].label])

with open(os.path.join(PATH_LABEL_VALID, 'labels_valid.csv'), 'w', newline='') as f:
    writer = csv.writer(f) 
    for img in valid_images:
        shutil.move(os.path.join(PATH_DATA, img), PATH_DATA_VALID)

        writer.writerow([img, labels.loc[img].label])        







