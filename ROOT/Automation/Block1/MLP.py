import torch
import torch.nn as nn
import torch.optim as optim

class MLPNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, 
                 number_of_hidden_neurons=100, 
                 number_of_output_neurons=3, 
                 number_of_layers=2, 
                 learning_rate=0.001, 
                 dropout_rate=0.25, 
                 seed_value=16981):
        """
        Initializes a multi-layer perceptron (MLP) network.
        
        Args:
            number_of_input_neurons (int): Dimensionality of input.
            number_of_hidden_neurons (int): Number of neurons in each hidden layer.
            number_of_output_neurons (int): Dimensionality of output.
            number_of_layers (int): Number of hidden layers.
            learning_rate (float): Learning rate for training.
            dropout_rate (float): Dropout rate.
            seed_value (int): Seed for reproducibility.
        """
        super(MLPNetwork, self).__init__()
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate

        layers = []
        input_size = number_of_input_neurons
        # Create the specified number of hidden layers
        for _ in range(number_of_layers):
            layers.append(nn.Linear(input_size, number_of_hidden_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = number_of_hidden_neurons

        # Output layer
        layers.append(nn.Linear(number_of_hidden_neurons, number_of_output_neurons))
        self.model = nn.Sequential(*layers)

        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output of the network.
        """
        x = x.to(self.device)
        return self.model(x)
    
    def train_model(self, train_data, target_data):
        """
        A simple training step using the Adam optimizer and MSE loss.
        
        Args:
            train_data (torch.Tensor): Input training data.
            target_data (torch.Tensor): Target output data.
            
        Returns:
            float: Loss value.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        prediction = self.forward(train_data)
        loss = criterion(prediction, target_data)
        loss.backward()
        optimizer.step()

        return loss.item()
