import torch.nn as nn
import torch

def train(model, iterator, optimizer, criterion, scheduler):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad() 
        torch.cuda.empty_cache()  
        text = batch.text
        label = batch.label
        predictions = model(text)
        loss = criterion(predictions, label)
        acc = accuracy(predictions, label) # TODO:Think about loss function
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    pass

class CustomModel(nn.Module):
    def __init__(self, bert_model, output_dim):
        super().__init__()
        self.bert_model = bert_model
        embedding_dim = bert_model.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.bert(text)[1]
        #embedded = [batch size, emb dim]
        output = self.out(embedded)
        #output = [batch size, out dim]


