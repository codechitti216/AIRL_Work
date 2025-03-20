import torch
import torch.nn as nn

class MemoryNeuralNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=100, number_of_output_neurons=3, 
                 learning_rate=0.001, learning_rate_2=0.0005, 
                 dropout_rate=0.25, lipschitz_constant=1.2, spectral_norm=False, seed_value=16981):
        super(MemoryNeuralNetwork, self).__init__()
        torch.manual_seed(seed_value)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons

        self.learning_rate = learning_rate
        self.learning_rate_2 = learning_rate_2
        self.lipschitz_constant = lipschitz_constant
        self.spectral_norm = spectral_norm

        self.alpha_input_layer = nn.Parameter(torch.rand(self.number_of_input_neurons, device=self.device))
        self.alpha_hidden_layer = nn.Parameter(torch.rand(self.number_of_hidden_neurons, device=self.device))
        self.alpha_last_layer = nn.Parameter(torch.rand(self.number_of_output_neurons, device=self.device))
        self.beta = nn.Parameter(torch.rand(self.number_of_output_neurons, device=self.device))

        self.weights_input_to_hidden_nn = nn.Parameter(torch.empty(
            self.number_of_input_neurons, self.number_of_hidden_neurons, device=self.device))
        self.weights_hidden_to_output_nn = nn.Parameter(torch.empty(
            self.number_of_hidden_neurons, self.number_of_output_neurons, device=self.device))
        self.weights_input_to_hidden_mn = nn.Parameter(torch.empty(
            self.number_of_input_neurons, self.number_of_hidden_neurons, device=self.device))
        self.weights_hidden_to_output_mn = nn.Parameter(torch.empty(
            self.number_of_hidden_neurons, self.number_of_output_neurons, device=self.device))

        torch.nn.init.xavier_uniform_(self.weights_input_to_hidden_nn)
        torch.nn.init.xavier_uniform_(self.weights_hidden_to_output_nn)
        torch.nn.init.xavier_uniform_(self.weights_input_to_hidden_mn)
        torch.nn.init.xavier_uniform_(self.weights_hidden_to_output_mn)
        
        self.dropout = nn.Dropout(dropout_rate)

        self.prev_output_of_nn = torch.zeros(self.number_of_output_neurons, device=self.device)
        
        self.to(self.device)

    def activation_function(self, x):
        return 15 * torch.tanh(x / 15)

    def forward(self, input_array):
        self.input_nn = input_array.clone().detach().to(self.device).requires_grad_(True)
        
        self.output_of_hidden_layer_nn = self.activation_function(
            torch.matmul(self.input_nn, self.weights_input_to_hidden_nn.clone()) +
            torch.matmul(self.input_nn, self.weights_input_to_hidden_mn.clone())
        )
        self.output_of_hidden_layer_nn = self.dropout(self.output_of_hidden_layer_nn)
        
        self.input_to_last_layer_nn = (
            torch.matmul(self.output_of_hidden_layer_nn, self.weights_hidden_to_output_nn.clone()) +
            torch.matmul(self.output_of_hidden_layer_nn, self.weights_hidden_to_output_mn.clone()) +
            (self.beta.clone() * self.prev_output_of_nn.clone())
        )
        
        self.output_nn = self.input_to_last_layer_nn
        
        self.prev_output_of_nn = self.output_nn.clone()
        
        return self.output_nn

    def backprop(self, y_des):
        y_des = torch.tensor(y_des, dtype=torch.float32, device=self.device)
        
        self.error_last_layer = self.output_nn - y_des
        
        with torch.no_grad():
            self.weights_hidden_to_output_nn -= self.learning_rate * torch.matmul(
                self.output_of_hidden_layer_nn.unsqueeze(1), self.error_last_layer.unsqueeze(0)
            )
            self.weights_input_to_hidden_nn -= self.learning_rate * torch.matmul(
                self.input_nn.unsqueeze(1), self.error_last_layer.unsqueeze(0)
            )

        return self.error_last_layer
