import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.io import read_image
from utils.dataset import VccorpDataset


def make_dataloader(input_size, batch_size):
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

    # target_transform = transforms.Lambda(lambda y: torch.zeros(
    #     2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))


    # PATH_DATA = './Data/data'
    # PATH_LABEL = './Data/label'

    # # make train, valid, test dataset
    # train_dataset = VccorpDataset(
    #     annotations_file= os.path.join(PATH_LABEL,'train', 'labels_train.csv'), 
    #     img_dir= os.path.join(PATH_DATA, 'train'), 
    #     transform=data_transforms['train'], 
    #     target_transform=target_transform
    #     )

    # valid_dataset = VccorpDataset(
    #     annotations_file= os.path.join(PATH_LABEL,'valid', 'labels_valid.csv'), 
    #     img_dir= os.path.join(PATH_DATA, 'valid'), 
    #     transform=data_transforms['val'], 
    #     target_transform=target_transform
    #     )

    # test_dataset = VccorpDataset(
    #     annotations_file= os.path.join(PATH_LABEL,'test', 'labels_test.csv'), 
    #     img_dir= os.path.join(PATH_DATA, 'test'), 
    #     transform=data_transforms['val'], 
    #     target_transform=target_transform
    #     )

    # # make train, valid, test dataloader

    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print("Initializing Datasets and Dataloaders...")

    data_dir = './Data/data'
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}

    # return train_dataloader, valid_dataloader, test_dataloader
    return dataloaders_dict




