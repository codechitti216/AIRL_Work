import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=100, 
                 number_of_output_neurons=3, dropout_rate=0.25, seed_value=16981):
        super(LSTMNetwork, self).__init__()
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lstm = nn.LSTM(input_size=number_of_input_neurons,
                            hidden_size=number_of_hidden_neurons,
                            batch_first=True)
        
        self.fc = nn.Linear(number_of_hidden_neurons, number_of_output_neurons)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.to(self.device)
    
    def forward(self, x):
        # x is expected to have shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # LSTM will create default hidden states if none provided.
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
