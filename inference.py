from importlib.resources import path
from tokenize import String
import yaml
import torch
import argparse
from torch import optim
from utils.utils import test_model, config_args_intersection, initialize_model, make_dataloader




def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", 
                        help="choose model to train", 
                        default="resnet")
    parser.add_argument("--batch_size", 
                        help="choose batch size", 
                        default=32, 
                        type=int)
    parser.add_argument("--feature_extract", 
                        help="choose fineturning or feature_extract", 
                        default=False, 
                        type=bool)
    parser.add_argument("--path_log", 
                        help="choose path of pretrained model", 
                        default='./log/resnet.pt', 
                        type=str)

    return parser



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)  

    parser = make_parser()
    args = parser.parse_args()
    # intersection config and args
    config = config_args_intersection(config, args)

    #init model
    model_ft, input_size = initialize_model(model_name=config['model'], num_classes=2, feature_extract=config["feature_extract"], use_pretrained=True)
    model_ft.load_state_dict(torch.load(config['path_log']))
    # make dataloader
    dataloaders_dict = make_dataloader(input_size=input_size, batch_size=config['batch_size'])
    
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Setup the loss fxn
    criterion = torch.nn.CrossEntropyLoss()

    # Test
    test_model(model_ft, dataloaders_dict, criterion, device)


    