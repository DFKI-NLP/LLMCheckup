from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
import random
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import sys
import pandas as pd
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

seed_val = 42
softmax = torch.nn.Softmax(dim=1)

def get_dataloader(data, batch_size, dtype):
    samples = []
    for i in range(len(data)):
        d_texts = data[i]['dialog']
        d_labels = data[i]['act']
        assert(len(d_texts)==len(d_labels))
        for i in range(len(d_texts)):
            samples.append((d_texts[i], d_labels[i]))
    dataset = DADataset(samples)
    if dtype=='train':# or dtype=='val':
        dataloader = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = batch_size, num_workers = 4)
    else:
        dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 1)
    return dataloader

class CustomDAModel:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   
    # predict:
    # X: array-like of shape (n_samples, n_features)
    # C: ndarray of shape (n_samples,)
    def predict(self, X):
        predictions = []
        model = self.model.to(self.device)
        #all_logits = np.zeros((len(X), model.output_classes))
        
        for b_input in X:
            print(b_input)
            b_input = self.tokenizer.encode_plus(b_input, return_tensors='pt')
            input_ids = b_input['input_ids'].to(self.device)
            input_mask = b_input['attention_mask'].to(self.device)

            model.zero_grad()
            with torch.no_grad():
                result = model(input_ids, input_mask)
                logits = result.detach().cpu().tolist() # result.logits...
                predictions.append(np.argmax(logits, axis=1)[0])
        predictions = np.asarray(predictions)
        return predictions

    # predict_proba(X)
    # X: array-like of shape (n_samples, n_features)
    # C: ndarray of shape (n_samples, n_classes)    
    def predict_proba(self, X):
        proba_predictions = []
        model = self.model.to(self.device)
        #all_logits = np.zeros((len(X), model.output_classes))
        
        for b_input in X:
            print(b_input)
            b_input = self.tokenizer.encode_plus(b_input, return_tensors='pt')
            input_ids = b_input['input_ids'].to(self.device)
            input_mask = b_input['attention_mask'].to(self.device)

            model.zero_grad()
            with torch.no_grad():
                result = model(input_ids, input_mask)
                probs = softmax(result).detach().cpu().tolist()[0] # result.logits...
                proba_predictions.append(probs)
        proba_predictions = np.asarray(proba_predictions)
        return proba_predictions
    

class DADataset:
    def __init__(self, samples):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.max_seq_length = 256
        self.input_ids = []
        self.input_masks = []
        self.labels = []
        
        for sample in samples:
            sample_text, label = sample
            ids, mask = self.get_id_with_mask(sample_text)
            self.input_ids.append(ids)
            self.input_masks.append(mask)
            self.labels.append(torch.tensor(label))

    def get_id_with_mask(self, input_text):
        encoded_dict = self.tokenizer.encode_plus(
                input_text.lower(),
                add_special_tokens = True, 
                max_length = self.max_seq_length,           
                pad_to_max_length = True,
                truncation=True, 
                return_attention_mask = True,   
                return_tensors = 'pt',     
           )
        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    def __getitem__(self, idx):
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        return self.input_ids[idx].squeeze(), self.input_masks[idx].squeeze(), label
        
    def __len__(self):
        return len(self.input_ids)
        
class DANetwork(nn.Module):
    def __init__(self):
        super(DANetwork, self).__init__()
        self.bert_emb_size = 768
        self.hidden_dim = 128
        self.output_classes = 5       
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.output_classes)
 
        
    def forward(self, input_ids, input_mask):
        output = self.bert(input_ids, attention_mask=input_mask).logits
        return output

def train(epochs, model, lr, criterion, optimizer, scheduler, train_dataloader, dev_dataloader, device):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    model = model.to(device)

    for epoch in range(epochs):
        
        total_train_loss = 0
        train_n_correct = 0
        
        total_dev_loss = 0
        dev_n_correct = 0
        best_avg_dev_loss = None

        model.train()

        for batch in tqdm(train_dataloader, 
                             total=len(train_dataloader),
                             desc=f'Train epoch {epoch+1}/{epochs}'):

            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()        
            result = model(input_ids, input_mask)
            loss = criterion(result, labels)
            loss.backward()
            total_train_loss += loss.item()
            
            optimizer.step()
            scheduler.step()

            logits = result.detach().cpu().numpy() # result.logits...
            label_ids = labels.to('cpu').numpy()
            _, _, _, accuracy = eval_result(logits, label_ids) 
            train_n_correct += accuracy
        
        avg_train_loss = total_train_loss / len(train_dataloader)            
        train_acc = train_n_correct / len(train_dataloader)


        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f} '.format(epoch+1, epochs, avg_train_loss, train_acc))
        
        with torch.no_grad():
            for batch in tqdm(dev_dataloader, 
                                 total=len(dev_dataloader),
                                 desc=f'Train epoch {epoch+1}/{epochs}'):

                input_ids = batch[0].to(device)
                input_mask = batch[1].to(device)
                labels = batch[2].to(device)

                model.zero_grad()        
                result = model(input_ids, input_mask)
                loss = criterion(result, labels)
                total_dev_loss += loss.item()
                
                logits = result.detach().cpu().numpy() # result.logits...
                label_ids = labels.to('cpu').numpy()
                _, _, _, accuracy = eval_result(logits, label_ids) 
                dev_n_correct += accuracy
            
            avg_dev_loss = total_dev_loss / len(dev_dataloader)            
            dev_acc = dev_n_correct / len(dev_dataloader)

            print('Epoch [{}/{}], Dev Loss: {:.4f}, Dev Accuracy: {:.4f} '.format(epoch+1, epochs, avg_dev_loss, dev_acc))
            if best_avg_dev_loss is None or (avg_dev_loss < best_avg_dev_loss):
                best_avg_dev_loss = avg_dev_loss
                print('Storing the model')
                torch.save(model.state_dict(), f"saved_model/{epoch}e_{lr}lr")

    print("Training complete!")

    torch.save(model.state_dict(), f"saved_model/{epochs}e_{lr}lr")

def eval_result(preds, labels):
    """ Calculate the accuracy, f1, precision, recall of our predictions vs labels
    """
    y_pred = np.argmax(preds, axis=1).flatten()
    y_true = labels.flatten()
    #print(y_pred[:50])
    #print(y_true[:50])
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(classification_report(np.asarray(y_true), np.asarray(y_pred)))
    print('precision:', round(precision,3), 'recall:', round(recall,3), 'f1:', round(f1,3), 'accuracy:', round(accuracy,3))
    return (precision, recall, f1, accuracy)

def evaluate(model, test_dataloader, device):
    model = model.to(device)
    all_logits = np.zeros((len(test_dataloader), model.output_classes))
    all_label_ids = np.zeros((len(test_dataloader), 1))
    for b_i, b_input in enumerate(test_dataloader):
        input_ids = b_input[0].to(device)
        input_mask = b_input[1].to(device)
        labels = b_input[2].to(device)

        model.zero_grad()
        with torch.no_grad():
            result = model(input_ids, input_mask)
            logits = result.detach().cpu().numpy() # result.logits...
            all_logits[b_i] = logits
            label_ids = labels.to('cpu').numpy()
            all_label_ids[b_i] = label_ids
    precision, recall, f1, accuracy = eval_result(all_logits, all_label_ids) 
    
def save_file(dataset, fname):
    lines = []
    idx = 0
    for el in dataset:
         dialog_context = ' '.join(el['dialog']).strip()
         for i, turn in enumerate(el['dialog']):
             lines.append(','.join([str(idx), turn.strip(), dialog_context, str(el['act'][i])]))
             idx+=1
    with open(fname, 'w') as f:
        f.write('id,turn,dialog_context,y\n')
        for line in lines:
            f.write(line+'\n')

def save_csv(dataset, fname):
    df = pd.DataFrame(dataset)
    df.to_csv(fname) #, index=False)

### code execution ###

epochs = 5
batch_size = 16
model = DANetwork()
lr = 5e-6
prepare_data_from_scratch = False
do_train = False

if prepare_data_from_scratch:
    train_data = load_dataset('daily_dialog', split='train')
    val_data = load_dataset('daily_dialog', split='validation')
    test_data = load_dataset('daily_dialog', split='test')
    train_data = train_data.select(range(1000))
    val_data = val_data.select(range(100))
    test_data = test_data.select(range(300))
    print('Test:', test_data)
    
    # save as separate files: [id, turn, dialog_context, da_label]
    save_csv(train_data, 'dataset_da_train.csv')
    save_csv(val_data, 'dataset_da_val.csv')
    save_csv(test_data, 'dataset_da_test.csv')
    
    train_dataloader = get_dataloader(train_data, batch_size, 'train')
    val_dataloader = get_dataloader(val_data, batch_size, 'val')
    test_dataloader = get_dataloader(test_data, 1, 'test')

    torch.save(train_dataloader, 'train_dataloader.pth')
    torch.save(val_dataloader, 'val_dataloader.pth')
    torch.save(test_dataloader, 'test_dataloader.pth')
else:
    train_dataloader = torch.load('train_dataloader.pth')
    val_dataloader = torch.load('val_dataloader.pth')
    test_dataloader = torch.load('test_dataloader.pth')


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
if do_train:
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),lr = lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    # training
    train(epochs, model, lr, criterion, optimizer, scheduler, train_dataloader, val_dataloader, device)
else:
    model.load_state_dict(torch.load('saved_model/5e_5e-06lr')) 

# evaluation on the test set
evaluate(model, test_dataloader, device)

formatted_model = CustomDAModel(model, device)
test_input = ['What do you mean?','This is interesting!']
formatted_model.predict_proba(test_input)
#torch.save(formatted_model, "da_model.pth")

# da_model.pkl
# .predict(X) and .predict_proba(X)
# predict:
# X: array-like of shape (n_samples, n_features)
# C: ndarray of shape (n_samples,)
# predict_proba(X)
# X: array-like of shape (n_samples, n_features)
# C: ndarray of shape (n_samples, n_classes)
