import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=100, number_of_output_neurons=3,
                 kernel_size=3, dilation_factor=2, dropout_rate=0.25, learning_rate=0.001, learning_rate_2=0.0005, 
                 lipschitz_constant=1.2, spectral_norm=False, seed_value=16981):
        super(TemporalConvolutionalNetwork, self).__init__()
        torch.manual_seed(seed_value)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons

        self.learning_rate = learning_rate
        self.learning_rate_2 = learning_rate_2
        self.lipschitz_constant = lipschitz_constant
        self.spectral_norm = spectral_norm

        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor

        self.alpha_input_layer = nn.Parameter(torch.rand(self.number_of_input_neurons, device=self.device))
        self.alpha_hidden_layer = nn.Parameter(torch.rand(self.number_of_hidden_neurons, device=self.device))
        self.alpha_last_layer = nn.Parameter(torch.rand(self.number_of_output_neurons, device=self.device))
        self.beta = nn.Parameter(torch.rand(self.number_of_output_neurons, device=self.device))

        self.conv1 = nn.Conv1d(self.number_of_input_neurons, self.number_of_hidden_neurons, kernel_size=self.kernel_size,
                               dilation=self.dilation_factor, padding=self.kernel_size-1, bias=False)
        self.conv2 = nn.Conv1d(self.number_of_hidden_neurons, self.number_of_output_neurons, kernel_size=self.kernel_size,
                               dilation=self.dilation_factor, padding=self.kernel_size-1, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        # Initialize prev_output_of_tcn with the correct shape
        self.prev_output_of_tcn = torch.zeros((1, self.number_of_output_neurons, 1), device=self.device)

        self.to(self.device)

    def activation_function(self, x):
        return 15 * torch.tanh(x / 15)

    def forward(self, input_array):
        self.input_tcn = input_array.to(self.device)

        x = self.conv1(self.input_tcn)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Use out-of-place operation instead of in-place
        x = x + self.beta.view(1, -1, 1) * self.prev_output_of_tcn.detach()


        self.prev_output_of_tcn = x.clone()  # Detach to prevent gradient accumulation

        return x

    def compute_loss(self, y_des):
        y_des = torch.tensor(y_des, dtype=torch.float32, device=self.device)
        loss = F.mse_loss(x, y_des)
        return loss  # Return the loss instead of calling .backward()