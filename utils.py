import json
import glob

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
        acc = categorical_accuracy(predictions, label)
        #torch.nn.utils.clip_grad_norm_(optimizer, max_grad_norm)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def parse_devset(path):
    all_files = glob.glob(path)