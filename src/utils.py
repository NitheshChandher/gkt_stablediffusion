import torch
import numpy as np
import matplotlib.pyplot as plt
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_score = np.inf

    def __call__(self, validation_loss):
        score = validation_loss
        if score > self.best_score - self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

def output_to_label(z):
    
    """
    Map network output z to a hard label {0, 1} 
    Args:
        z (Tensor): Probabilities for each sample in a batch.
    Returns:
        c (Tensor): Hard label {0, 1} for each sample in a batch
    """
    c = (z > 0.5).type(torch.long)
    return c

def viz(train_losses, train_acc, val_losses, val_acc, path, name):
    """
    Plot performance curves of the models and saves the plot as PNG
    
    Args: 
        - train_losses (list): List of training loss values
        - train_acc (list): List of training accuracy values
        - val_losses (list): List of validation loss values
        - val_acc (list): List of validation accuracy values
        - path (str): Path to save the plot as a PNG
    """
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(12, 6))
    #Plot Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, 'b', label='Train Loss')
    plt.plot(epochs, val_losses, 'r', label = 'Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    #Plot Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, 'b', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(path+name+'.png')
