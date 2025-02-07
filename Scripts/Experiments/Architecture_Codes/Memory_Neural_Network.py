import torch
import torch.nn as nn

class MemoryNeuralNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=60, number_of_output_neurons=3, neeta=1.2e-3, neeta_dash=5e-4, lipschitz_norm=1.2, spectral_norm=False, dropout_rate=0.0, seed_value=16981):
        super(MemoryNeuralNetwork, self).__init__()

        torch.manual_seed(seed_value)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons

        self.spectral_norm = spectral_norm
        self.lipschitz = lipschitz_norm
        self.dropout_rate = dropout_rate

        self.neeta = neeta
        self.neeta_dash = neeta_dash

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
        
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        self.prev_output_of_input_layer_nn = torch.zeros(self.number_of_input_neurons, device=self.device)
        self.prev_output_of_input_layer_mn = torch.zeros(self.number_of_input_neurons, device=self.device)
        self.prev_output_of_hidden_layer_nn = torch.zeros(self.number_of_hidden_neurons, device=self.device)
        self.prev_output_of_hidden_layer_mn = torch.zeros(self.number_of_hidden_neurons, device=self.device)
        self.prev_output_of_nn = torch.zeros(self.number_of_output_neurons, device=self.device)
        self.prev_output_of_mn = torch.zeros(self.number_of_output_neurons, device=self.device)
        
        self.prev_error_hidden_layer = torch.zeros(self.number_of_hidden_neurons, device=self.device)

        self.to(self.device)

    def activation_function(self, x):
        return 15 * torch.tanh(x / 15)

    def activation_function_derivative(self, x):
        return 1.0 - torch.tanh(x / 15) ** 2

    def feedforward(self, input_array):
        self.input_nn = torch.tensor(input_array, dtype=torch.float32, device=self.device)

        self.output_of_input_layer_nn = self.input_nn
        self.output_of_input_layer_mn = (self.alpha_input_layer * self.prev_output_of_input_layer_nn +
                                         (1.0 - self.alpha_input_layer) * self.prev_output_of_input_layer_mn)

        self.input_to_hidden_layer_nn = torch.matmul(self.output_of_input_layer_nn, self.weights_input_to_hidden_nn) + \
                                        torch.matmul(self.output_of_input_layer_mn, self.weights_input_to_hidden_mn)
        self.output_of_hidden_layer_nn = self.activation_function(self.input_to_hidden_layer_nn)
        self.output_of_hidden_layer_nn = self.dropout(self.output_of_hidden_layer_nn)
        self.output_of_hidden_layer_mn = (self.alpha_hidden_layer * self.prev_output_of_hidden_layer_nn +
                                          (1.0 - self.alpha_hidden_layer) * self.prev_output_of_hidden_layer_mn)

        self.input_to_last_layer_nn = torch.matmul(self.output_of_hidden_layer_nn, self.weights_hidden_to_output_nn) + \
                                      torch.matmul(self.output_of_hidden_layer_mn, self.weights_hidden_to_output_mn) + \
                                      (self.beta * self.prev_output_of_nn)

        self.output_nn = self.input_to_last_layer_nn

        self.prev_output_of_input_layer_nn = self.output_of_input_layer_nn.clone()
        self.prev_output_of_input_layer_mn = self.output_of_input_layer_mn.clone()
        self.prev_output_of_hidden_layer_nn = self.output_of_hidden_layer_nn.clone()
        self.prev_output_of_hidden_layer_mn = self.output_of_hidden_layer_mn.clone()
        self.prev_output_of_nn = self.output_nn.clone()

        return self.output_nn
