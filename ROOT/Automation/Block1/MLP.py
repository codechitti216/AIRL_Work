import torch
import torch.nn as nn
import torch.optim as optim

class MLPNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=100, 
                 number_of_output_neurons=3, number_of_layers=2, learning_rate=0.001, dropout_rate=0.25, seed_value=16981):
        super(MLPNetwork, self).__init__()
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = []
        input_size = number_of_input_neurons

        for _ in range(number_of_layers):
            layers.append(nn.Linear(input_size, number_of_hidden_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = number_of_hidden_neurons
        
        layers.append(nn.Linear(number_of_hidden_neurons, number_of_output_neurons))
        self.model = nn.Sequential(*layers)

        self.learning_rate = learning_rate
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)
    
    def train_model(self, train_data, target_data):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        prediction = self.forward(train_data)
        loss = criterion(prediction, target_data)
        loss.backward()
        optimizer.step()

        return loss.item()
