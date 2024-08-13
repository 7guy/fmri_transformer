import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from torch import nn,optim 
from preprocess_TEC import get_dataloaders
from preprocess2 import get_dataloaders2
from preprocess2 import testDict, matrixList
import numpy as np

from models.model.transformer import Transformer

def cross_entropy_loss(pred, target): 
    criterion = nn.CrossEntropyLoss() 
    lossClass= criterion(pred, target )
    return lossClass 

def calc_loss_and_score(pred, target, metrics): 
    softmax = nn.Softmax(dim=1)

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1).long()
    
    ce_loss = cross_entropy_loss(pred, target) 

    metrics['loss'] .append( ce_loss.item() )
    pred = softmax(pred ) 
    _,pred = torch.max(pred, dim=1) 
    correct = torch.sum(pred ==target ).item() 
    metrics['correct']  += correct
    total = target.size(0)   
    metrics['total']  += total
    print('loss : ' +str(ce_loss.item() ) + ' correct: ' + str(((100 * correct )/total))  + ' target: ' + str(target.data.cpu().numpy()) + ' prediction: ' + str(pred.data.cpu().numpy()))
    return ce_loss

def print_average(metrics):  

    loss= metrics['loss'] 
    score = 100 * (metrics['correct'] / metrics['total'])
    print('average loss : ' +str(np.mean(loss))  + 'average correct: ' + str(score))
    return score, loss
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def test_model(model,test_loader,device):
    model.eval() 
    metrics = dict()
    metrics['loss']=list()
    metrics['correct']=0
    metrics['total']=0

    all_preds = []
    all_targets = []

    for inputs, labels in test_loader:
        with torch.no_grad():
            
            inputs = inputs.to(device=device, dtype=torch.float )
            labels = labels.to(device=device, dtype=torch.int) 
            pred = model(inputs)

            calc_loss_and_score(pred, labels, metrics)

            _, predicted = torch.max(pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    score, loss = print_average(metrics)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(conf_matrix)

    return score, loss

data_directory = r"D:\Final Project\TASK_RH_vis2\dataset"
NET = 'Default_pCunPCC'
NET_idx = 3
#NET = 'Vis'
#NET = 'DorsAttn_Post'
H= 'RH'
batch_size = 1
#
device = torch.device("cuda")
sequence_len=65 # sequence length of time series
max_len=65 # max time series sequence length
n_head = 1 # number of attention head
n_layer = 1 # number of encoder layer
drop_prob = 0.1
d_model = 512 # number of dimension ( for positional embedding)
ffn_hidden = 128 # size of hidden layer before classification
details = False
lr = 0.0001
num_of_epoches = 40
#
# dataloaders, voxels = get_dataloaders(NET, NET_idx, H, batch_size, sequence_len)


dataloaders, voxels = get_dataloaders2(data_directory, NET, NET_idx, H, slice='end', batch_size=batch_size)
#
print(len(testDict))
feature = voxels # for univariate time series (1d), it must be adjusted for 1.
test_dataloader = dataloaders['test']
#
model =  Transformer(voxels =voxels, d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=details,device=device).to(device=device)


# model.load_state_dict(torch.load('myModel2'))
model.load_state_dict(torch.load('saved_models/Model_RH_PCC_3_last65noshuffle'))
test_model(device=device, model=model, test_loader=test_dataloader)

print(len(matrixList))
dir = './saved_models'
dir2 = './TimeSeriesProject'
output_dir = './saved_plots'
scores_dict = {}

# ==== Post test - result analysis - Get attention matrices  ====
# Step 1: Initialize 
testDict = {f'subject{i+1}': [] for i in range(18)}
# Step 2: Distribute matrices from matrixList into testDict
for i, subject in enumerate(testDict.keys()):
    start_idx = i * 14
    end_idx = start_idx + 14
    testDict[subject] = matrixList[start_idx:end_idx]

# to get RDMs, go to preprocess2 file and turn on the svm and then:
corr_matrices = [np.corrcoef(matrix) for matrix in test_dict['input']]
subjects = {f'subject{i+1}': [] for i in range(18)
for i, subject in enumerate(subjects.keys()):
    start_idx = i * 14
    end_idx = start_idx + 14
    subjects[subject] = corr_matrices[start_idx:end_idx]

# save the relevant matrices you want as .npy files and move to the analysis on the rdm/attention on google colab notebook (Link on the README)
