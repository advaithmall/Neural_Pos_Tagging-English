import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import LSTMTagger
from dataset import PosDataset
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device: ", device)

def train(dataset, model, args):
    print("Entered Training...")
    count=0
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    loss_function = nn.NLLLoss()
    accuracy_list = list()
    recall_list = list()
    precision_list = list()
    f1_list = list()
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    for epoch in range(args.max_epochs):
        for batch, (sentence, tags) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()
            y_pred = model(sentence).to(device)
            loss = loss_function(y_pred, tags)
            loss.backward()
            optimizer.step()
            pred_list = list()
            for i in range(len(y_pred)):
                pred_list.append(y_pred[i].argmax().item())
            accuary = (y_pred.argmax(1) == tags).float().mean()
            tags = tags.tolist()
            pred_list = pred_list
            recall = recall_score(tags, pred_list, average='macro', zero_division=0)
            precision = precision_score(tags, pred_list, average='macro', zero_division=0)
            f1 = f1_score(tags, pred_list, average='macro', zero_division=0)
            accuracy_list.append(accuary.item())
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
            print({
                'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'acc': accuary.item(),'f1': f1})
    print("avg acc: ", sum(accuracy_list)/len(accuracy_list), "avg f1: ", sum(f1_list)/len(f1_list))
              
def eval(args, model, val_dataset):
    print("entered eval")
    model.eval()
    avg_list = list()
    precision_list = list()
    recall_list = list()
    f1_list = list()
    loss_function = nn.NLLLoss()
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    state_h, state_c = model.init_state(args.sequence_length)
    for batch, (sentence, val_tags) in enumerate(val_dataloader):
            y_pred = model(sentence)
            pred_list = list()
            accuary = (y_pred.argmax(1) == val_tags).float().mean()
            for i in range(len(y_pred)):
                pred_list.append(y_pred[i].argmax().item())
            avg_list.append(accuary.item())
            val_tags = val_tags.tolist()
            pred_list = pred_list   
            recall = recall_score(val_tags, pred_list, average='macro', zero_division=0)
            precision = precision_score(val_tags, pred_list, average='macro', zero_division=0)
            f1 = f1_score(val_tags, pred_list, average='macro', zero_division=0)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            print("validation: ", { 'batch': batch, 'acc': accuary.item(), 'recall': recall, 'precision': precision, 'f1': f1})
    print("avg acc: ", sum(avg_list)/len(avg_list), "avg recall: ", sum(recall_list)/len(recall_list), "avg precision: ", sum(precision_list)/len(precision_list), "avg f1: ", sum(f1_list)/len(f1_list))
        
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="train")
parser.add_argument('--max-epochs', type=int, default=25)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()


dataset = torch.load("dataset.pt")
model = torch.load("model.pt")
model = model.to(device)
test_dataset = torch.load("test_dataset.pt")
#eval(args, model, test_dataset)
val_dataset = torch.load("val_dataset.pt")
#eval(args, model, val_dataset)

while(1):
        input_1 = input("Enter a Sentence: ")
        input_1 = re.sub(r'[^\w\s]','',input_1)
        input_x = input_1.lower()
        input_1 = list()
        #append 3 padding tokens to input_1
        input_1.append("<pad>")
        input_1.append("<pad>")
        input_1.append("<pad>")
        input_y = input_x.split()
        for word in input_y:
            input_1.append(word)
        len_x = len(input_1)
        word_in = input_1
        input_1 = [dataset.word_to_index.get(word, dataset.word_to_index["<unk>"]) for word in input_1]
        input_1 = torch.tensor(input_1)
        #print(input_1)
        for i in range(0, len_x-3):
            #print("i: ", i)
            #print(input_1[i:i+4])
            x = input_1[i:i+4].to(device)
            x = x.view(1,4)
            #print(x.shape)
            y_pred = model(x).to(device)
            #get argmax from y_pred
            pred = y_pred.argmax().item()
            #convert pred index to tag
            tag = dataset.index_to_tag[pred]
            print(word_in[i+3], "\t", tag)
