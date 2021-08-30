import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class LSTM_V1(nn.Module):
    def __init__(self):
        super(LSTM_V1, self).__init__()
        self.LSTM = nn.LSTM(input_size = 16, hidden_size=768, num_layers=1, batch_first=True)
        self.fc = nn.Linear(768, 3)

    def forward(self, x, x_lens):
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False) # [B, S, D]
        output, _ = self.LSTM(x)
        output, output_lens = pad_packed_sequence(output, batch_first=True)
        output = self.fc(output)
        return output

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(35, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 20)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTM_V2(nn.Module):
    def __init__(self):#, #input_dim):
        super(LSTM_V2, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=128,
                            num_layers=1, batch_first=True)      
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 20)              

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc2(self.fc1(output[:, -1, :]))