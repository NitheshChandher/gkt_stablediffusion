import torch
import argparse
import yaml
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomRotation, RandomHorizontalFlip
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from model import CNN
from utils import EarlyStopping, output_to_label, viz
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import dataframe_image as dfi

def model_setup(config,seed,exp):
    #=== Directories ===#
    if exp == 'AFHQ':
        exp_name = config['model1']
    
    else:
        exp_name = config['model2'] 

    save_path = config['save_path'] + f'{exp_name}-{seed}'
    save_plot = config['save_plot'] + f'{exp_name}-{seed}'

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    if not os.path.exists(save_plot):
        os.makedirs(save_plot)

    #=== Data Param ===#
    img_size = config['img_size']
    bs = config['batch_size']

    #=== Training Param ===#
    learn_rate = config['learning_rate']
    beta1 = config['beta_1']
    beta2 = config['beta_2']
    w_decay = config['weight_decay']
    epochs = config['num_epochs']

    if config['loss'] == 'BCE':
        loss = nn.BCELoss()
    
    elif config['loss']== 'NLL':
        loss = nn.NLLLoss()

    elif config['loss']== 'CrossEntropy':
        loss = nn.CrossEntropyLoss()

    else:
        print('The loss function is not defined. Choose BCE, NLL or CrossEntropy in config file') 
    
    #=== Data Loaders ===#
        
    #Define Model
    model = CNN(image_size=img_size)

    #Define Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = learn_rate,
                                  betas = (beta1,beta2),
                                  weight_decay=w_decay,
                                  eps=1e-08,)
    #Run the training loop
    model, train_losses, train_acc, val_losses, val_acc = train(model=model, optimizer=optimizer, 
                                                                loss_fn=loss, train_loader=train_dataloader,
                                                                val_loader=val_dataloader, num_epochs=epochs, 
                                                                config=config)
    torch.save(model, save_path)
    viz(train_losses, train_acc, val_losses, val_acc, save_plot)
    print(f'Model-{exp_name}-{seed} training is completed!')

def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, config):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    early_stopping = EarlyStopping(tolerance=config['tolerance'], min_delta=config['min_delta'])
    for epoch in range(1, num_epochs+1):
        model, train_loss, train_acc = train_epoch(model,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   device
                                                   )
        val_loss, val_acc = eval_epoch(model, loss_fn, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
              f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
              f"Val. loss: {sum(val_loss)/len(val_loss):.3f}, "
              f"Val. acc.: {sum(val_acc)/len(val_acc):.3f}")
        train_losses.append(sum(train_loss)/len(train_loss))
        train_accs.append(sum(train_acc)/len(train_acc))
        val_losses.append(sum(val_loss)/len(val_loss))
        val_accs.append(sum(val_acc)/len(val_acc))
        if config['early_stopping'] == 'yes':
            early_stopping(sum(train_loss)/len(train_loss),val_loss)
            if early_stopping.early_stop:
                print("Training stopped at:", epoch)
                break
    return model, train_losses, train_accs, val_losses, val_accs

def train_epoch(model, optimizer, loss_fn, train_loader, device):
    #Train:
    model.train()
    train_loss, train_acc = [], []
    for _, (x,y) in enumerate(train_loader,1):
        input, label = x.to(device), y.to(device)
        optimizer.zero_grad()
        z = model.forward(input)
        loss = loss_fn(z, label.float())
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        hard_pred = output_to_label(z)
        acc_avg = (hard_pred == label).float().mean().item()
        train_acc.append(acc_avg)

    return model, train_loss, train_acc

def eval_epoch(model, loss_fn, val_loader, device):
    eval_loss, eval_acc = [], []
    model.eval()
    with torch.no_grad():
        for _,(x,y) in enumerate(val_loader,1):
            input, label = x.to(device), y.to(device)
            z = model.forward(input)
            loss = loss_fn(z, label.float())
            eval_loss.append(loss)
            hard_pred = output_to_label(z)
            acc_avg = (hard_pred == label).float().mean().item()
            eval_acc.append(acc_avg) 
    return eval_loss, eval_acc

def setup(config):
    #=== Directories ===#
    rd_path = config['real_data_path']
    sd_path = config['syn_data_path']
    save_path = config['save_path']
    model1 = config['model1']
    model2 = config['model2']
    save_rmodel = save_path + model1
    save_synmodel =  save_path + model2
    seeds = config['seed']
    seed = seeds[0]
    model_setup(config,seed,exp=model1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()
    with open(args.config,"r") as file:
        config = yaml.safe_load(file)
    setup(config)

if __name__ == '__main__':
    main()