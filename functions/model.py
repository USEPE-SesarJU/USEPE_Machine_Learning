import torch
import torch.nn as nn

class myRNN(nn.Module):
    def __init__(self, input=2, hidden=64, output=2):
        super().__init__()
        self.input_num = input
        self.hidden_num = hidden
        self.n_layers = 1

        self.rnn = nn.RNN(input, hidden, self.n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        inith = self.init_hiddens(x.size(0), x.device)
        x_rnn, hidden = self.rnn(x, inith)
        out = self.fc(x_rnn[:, -1, :])
        return out

    def init_hiddens(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_num).to(device)