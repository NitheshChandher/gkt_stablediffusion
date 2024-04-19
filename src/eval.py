import torch
import argparse
import yaml
import os
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from dataset import makedataset
from utils import output_to_label

def eval(config):
    data_path = config['data_path']
    model_path = config['model_path']
    bs = config['batch_size']
    img_size = config['img_size']
    test_dataloader = makedataset(data_path+f'test',img_size,bs,'Test')
    for _model_ in os.listdir(model_path):
        model = torch.load(model_path + _model_)
        device = torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")
        model.to(device)
        model.eval()
        y_pred, y_true = [],[]
        results = pd.DataFrame(columns=['Model','Accuracy','Precision','Recall','F-Score'])
        with torch.no_grad():
            for _, (x,y) in enumerate(test_dataloader,1):
                input, label = x.to(device), y.to(device)
                pred = model(input)
                pred = output_to_label(pred)
                y_pred.append(pred.cpu().numpy())
                y_true.append(label.cpu().numpy())
        y_pred =  list(np.concatenate(y_pred))
        y_true =  list(np.concatenate(y_true))
        precision = precision_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred)
        f_score = f1_score(y_true,y_pred)
        accuracy = accuracy_score(y_true,y_pred)
        test_result = {'Model':_model_, 'Accuracy':accuracy, 'Precision':precision, 
                       'Recall':recall, 'F-Score':f_score}
        df = pd.DataFrame(test_result,index=[0])
        results = results.append(df, ignore_index=True)
    return results
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()
    with open(args.config,"r") as file:
        config = yaml.safe_load(file)
    test_results = eval(config)
    print(test_results)

if __name__ =='__main__':
    main()