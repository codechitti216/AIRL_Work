import torch
import torch.nn as nn

class FourierAnalysisNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=100, 
                 number_of_output_neurons=4, dropout_rate=0.25, seed_value=16981):
        super(FourierAnalysisNetwork, self).__init__()
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Using a real FFT (rfft) on a real-valued input of length N gives N//2+1 frequency bins.
        self.fft_feature_size = number_of_input_neurons // 2 + 1
        
        # Fully connected layers on the frequency-domain features.
        self.fc1 = nn.Linear(self.fft_feature_size, number_of_hidden_neurons)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(number_of_hidden_neurons, number_of_output_neurons)
        
        self.to(self.device)
    
    def forward(self, x):
        # x is expected to have shape: (batch, input_size)
        x = x.to(self.device)
        # Compute FFT along the last dimension using rfft
        fft_out = torch.fft.rfft(x)
        # Use magnitude (absolute value) as features
        fft_features = torch.abs(fft_out)
        hidden = self.relu(self.fc1(fft_features))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        return output
