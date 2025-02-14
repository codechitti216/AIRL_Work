import torch
import torch.nn as nn
import torch.optim as optim

class LSTMNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=100, 
                 number_of_output_neurons=3, learning_rate=0.001, dropout_rate=0.25, seed_value=16981):
        super(LSTMNetwork, self).__init__()
        
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lstm = nn.LSTM(input_size=number_of_input_neurons,
                            hidden_size=number_of_hidden_neurons,
                            batch_first=True)
        
        self.fc = nn.Linear(number_of_hidden_neurons, number_of_output_neurons)
        self.dropout = nn.Dropout(dropout_rate)
        self.learning_rate = learning_rate
        
        self.to(self.device)
    
    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        h_0, c_0 = (torch.zeros(1, 1, self.lstm.hidden_size).to(self.device),
                    torch.zeros(1, 1, self.lstm.hidden_size).to(self.device))
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(0)
    
    def train_model(self, train_data, target_data, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        optimizer.zero_grad()
        prediction = self.forward(train_data)
        loss = criterion(prediction, target_data)
        loss.backward()
        optimizer.step()
        
        return loss.item()
