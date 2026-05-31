import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle as pkl
import matplotlib.pyplot as plt

class MusicLSTM(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers,droupout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            dropout=droupout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)

    def forward(self,X,hidden=None):
       embedding = self.embed(X)
       output, hidden = self.lstm(embedding,hidden)
       logits = self.linear(output)

       return logits,hidden
       

class MusicDataset(Dataset):
    def __init__(self,data,seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data)-1)//self.seq_len
    
    def __getitem__(self, index):
        start_indxx = self.seq_len*(index)
        return (self.data[start_indxx:start_indxx+self.seq_len],
                self.data[start_indxx+1:start_indxx+self.seq_len+1])



with open('seq_joined.pkl','rb') as f:
    joined_seq = pkl.load(f)

train_size = 0.9
split_index = int(0.9*len(joined_seq))

train_data = joined_seq[:split_index]
test_data = joined_seq[split_index:]

train_data = torch.tensor(train_data)
test_data = torch.tensor(test_data)

train_data = MusicDataset(train_data,128)
test_data = MusicDataset(test_data,128)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

with open('itos.pkl','rb') as f:
    itos = pkl.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(itos)

Model = MusicLSTM(vocab_size,64,256,2,0.4).to(device)

loss_f = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(Model.parameters(),lr = 0.001)

epochs = 20

train_losses = []
test_losses = []

for _ in range(epochs):
    train_loss = 0

    for X,Y in train_dataloader:
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        logits,_ = Model(X)
        loss = loss_f(logits.view(-1, vocab_size), Y.view(-1))
        train_loss+= loss.item()
        loss.backward()
        optimizer.step()
    train_loss = train_loss/len(train_dataloader)
    train_losses.append(train_loss)
    print(f'{_}: {train_loss}')

    test_loss = 0
    Model.eval()
    with torch.no_grad():
        for X,Y in test_dataloader:
            X = X.to(device)
            Y = Y.to(device)
            logits,_ = Model(X)
            loss = loss_f(logits.view(-1, vocab_size), Y.view(-1))
            test_loss+= loss.item()
        test_loss = test_loss/len(test_dataloader)
        test_losses.append(test_loss)
    
    Model.train()

    
plt.plot(range(epochs),train_losses,label = 'train_loss')
plt.plot(range(epochs),test_losses,label = 'test_loss')
plt.legend()
plt.show()
   
torch.save(Model.state_dict(), 'music_lstm.pth')