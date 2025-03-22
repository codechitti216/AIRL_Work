import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out[:, :, :x.size(2)])
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out[:, :, :x.size(2)])
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, num_layers=3, dilation_base=2, num_output_neurons=4, seed_value=16981):
        super(TCN, self).__init__()
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = []
        in_channels = 1  # treat input as single-channel
        for i in range(num_layers):
            out_channels = num_channels
            dilation = dilation_base ** i
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels * num_inputs, num_output_neurons)
        self.to(self.device)
    
    def forward(self, x):
        # x shape: (batch, seq_len) -> reshape to (batch, 1, seq_len)
        x = x.unsqueeze(1)
        y = self.network(x)
        y = y.view(y.size(0), -1)
        return self.fc(y)
