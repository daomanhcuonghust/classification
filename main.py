import yaml
import torch
import argparse
from torch import optim
from utils.utils import train_model, config_args_intersection, initialize_model, make_dataloader




def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", 
                        help="choose model to train", 
                        default="resnet")
    parser.add_argument("--lr", 
                        help="choose learning rate ", 
                        default=1e-4, 
                        type=float)
    parser.add_argument("--batch_size", 
                        help="choose batch size", 
                        default=32, 
                        type=int)
    parser.add_argument("--num_epochs", 
                        help="choose number of epochs", 
                        default=50, 
                        type=int)
    parser.add_argument("--feature_extract", 
                        help="choose fineturning or feature_extract", 
                        default=False, 
                        type=bool)

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
    
    # make dataloader
    dataloaders_dict = make_dataloader(input_size=input_size, batch_size=config['batch_size'])
    
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if config["feature_extract"]:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    import pdb;pdb.set_trace()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=config["lr"])

    # Setup the loss fxn
    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    # model_ft, hist = train_model(model_ft, 
    #                             dataloaders_dict, 
    #                             criterion, 
    #                             optimizer_ft, 
    #                             num_epochs=config['num_epochs'], 
    #                             is_inception=(config['model']=="inception"),
    #                             device=device,
    #                             path='./log/checkpoint.pt')


    