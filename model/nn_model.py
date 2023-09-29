import torch
import torch.nn as nn

from model.nn_blocks import Att_Block, TemporalConvNet


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = x.unsqueeze(-1)
        return x


class Vanilla_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2):
        super(Vanilla_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        out, _ = self.lstm(x, (h0.detach(),c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.unsqueeze(-1)
        return out


class Att_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2):
        super(Att_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Att_Block(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.device)  # 初始化成0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        out, _ = self.lstm(x, (h0.detach(),c0.detach()))
        out = self.attention(out)
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.unsqueeze(-1)
        return out



class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels=[32, 32], kernel_size=3, dropout=0.3):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs = inputs.permute(0, 2, 1)
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        out = self.fc(out[:, :, -1])
        return out.unsqueeze(-1)

if __name__ == "__main__":
    print(1)