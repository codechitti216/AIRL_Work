import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMNetwork(nn.Module):
    def __init__(self, number_of_input_neurons, number_of_hidden_neurons,
                 number_of_output_neurons, neeta, seed_value=12345):  
        super(LSTMNetwork, self).__init__()
        
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lstm = nn.LSTM(input_size=number_of_input_neurons, 
                             hidden_size=number_of_hidden_neurons, 
                             batch_first=True)
        
        self.fc = nn.Linear(number_of_hidden_neurons, number_of_output_neurons)
        
        self.optimizer = optim.Adam(self.parameters(), lr=neeta)
        self.criterion = nn.MSELoss()
        
        self.hidden_state = None  
        
        self.neeta = neeta
        
        self.to(self.device)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  
        
        if self.hidden_state is None:
            h_0 = torch.zeros(1, 1, self.lstm.hidden_size).to(self.device)
            c_0 = torch.zeros(1, 1, self.lstm.hidden_size).to(self.device)
        else:
            h_0, c_0 = self.hidden_state
            h_0, c_0 = h_0.detach(), c_0.detach()  
        
        out, self.hidden_state = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  
        return out.squeeze(0)

    def reset_hidden_state(self):
        self.hidden_state = None  

    def backprop(self, x, target):
        self.optimizer.zero_grad()
        prediction = self.forward(x)
        loss = self.criterion(prediction, target.squeeze())  
        loss.backward()
        self.optimizer.step()
        return loss.item()
