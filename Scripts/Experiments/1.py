import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

np.random.seed(0)
torch.manual_seed(0)

import pandas as pd
import numpy as np
import torch

# Load dataset
df = pd.read_csv("../../Data/new_dataset.csv")

# Define input and output columns
input_columns = [
    'ACC X', 'ACC Y', 'ACC Z', 'GYRO X', 'GYRO Y', 'GYRO Z',
    'DVL0_1', 'DVL0_2', 'DVL0_3', 'DVL0_4',
    'DVL1_1', 'DVL1_2', 'DVL1_3', 'DVL1_4',
    'DVL2_1', 'DVL2_2', 'DVL2_3', 'DVL2_4',
    'DVL3_1', 'DVL3_2', 'DVL3_3', 'DVL3_4'
]
input_data = df[input_columns].values

output_columns = ['Vx', 'Vy', 'Vz']
target_data = df[output_columns].values

# Split dataset
num_samples = len(df)
train_samples = 2 * num_samples // 3  

X_train = input_data[:train_samples]
y_train = target_data[:train_samples]

X_test = input_data[train_samples:]
y_test = target_data[train_samples:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("Data successfully loaded and converted to tensors.")

class MemoryNeuralNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=60, number_of_output_neurons=3, neeta=1.2e-3, neeta_dash=5e-4, lipschitz_norm=1.2, spectral_norm=False, seed_value=16981):
        super(MemoryNeuralNetwork, self).__init__()

        torch.manual_seed(seed_value)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons

        self.spectral_norm = spectral_norm
        self.lipschitz = lipschitz_norm

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
        
        self.prev_output_of_input_layer_nn = torch.zeros(self.number_of_input_neurons, device=self.device)
        self.prev_output_of_input_layer_mn = torch.zeros(self.number_of_input_neurons, device=self.device)
        self.prev_output_of_hidden_layer_nn = torch.zeros(self.number_of_hidden_neurons, device=self.device)
        self.prev_output_of_hidden_layer_mn = torch.zeros(self.number_of_hidden_neurons, device=self.device)
        self.prev_output_of_nn = torch.zeros(self.number_of_output_neurons, device=self.device)
        self.prev_output_of_mn = torch.zeros(self.number_of_output_neurons, device=self.device)

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

    def backprop(self, y_des):
        y_des = torch.tensor(y_des, dtype=torch.float32, device=self.device)

        
        self.error_last_layer = self.output_nn - y_des
        self.error_hidden_layer = self.activation_function_derivative(self.input_to_hidden_layer_nn) * \
                                torch.matmul(self.error_last_layer, self.weights_hidden_to_output_nn.t())
        
        self.pd_e_wrt_v_input_layer = torch.matmul(self.error_hidden_layer, self.weights_input_to_hidden_nn.t())  
        self.pd_v_wrt_alpha_input_layer = self.output_of_input_layer_nn  

        self.pd_e_wrt_v_hidden_layer = self.error_hidden_layer
        self.pd_v_wrt_alpha_hidden_layer = self.output_of_hidden_layer_nn  

        self.pd_e_wrt_v_last_layer = self.error_last_layer
        self.pd_v_wrt_alpha_last_layer = self.output_nn  
        
        self.weights_hidden_to_output_nn.data -= self.neeta * self.error_last_layer.repeat(self.number_of_hidden_neurons, 1) * \
                                                self.output_of_hidden_layer_nn.repeat(self.number_of_output_neurons, 1).t()

        self.weights_input_to_hidden_nn.data -= self.neeta * self.error_hidden_layer.repeat(self.number_of_input_neurons, 1) * \
                                                self.output_of_input_layer_nn.repeat(self.number_of_hidden_neurons, 1).t()

        self.weights_hidden_to_output_mn.data -= self.neeta * self.error_last_layer.repeat(self.number_of_hidden_neurons, 1) * \
                                                self.output_of_hidden_layer_mn.repeat(self.number_of_output_neurons, 1).t()

        self.weights_input_to_hidden_mn.data -= self.neeta * self.error_hidden_layer.repeat(self.number_of_input_neurons, 1) * \
                                                self.output_of_input_layer_mn.repeat(self.number_of_hidden_neurons, 1).t()
        
        self.alpha_hidden_layer.data -= self.neeta_dash * self.pd_e_wrt_v_hidden_layer * self.pd_v_wrt_alpha_hidden_layer
        self.alpha_input_layer.data -= self.neeta_dash * self.pd_e_wrt_v_input_layer * self.pd_v_wrt_alpha_input_layer
        self.alpha_last_layer.data -= self.neeta_dash * self.pd_e_wrt_v_last_layer * self.pd_v_wrt_alpha_last_layer
        if self.spectral_norm:
            self.weights_input_to_hidden_mn.data = (self.weights_input_to_hidden_mn / torch.norm(self.weights_input_to_hidden_mn, p=2)) * \
                                                    (self.lipschitz ** (1 / 2))
            self.weights_input_to_hidden_nn.data = (self.weights_input_to_hidden_nn / torch.norm(self.weights_input_to_hidden_nn, p=2)) * \
                                                    (self.lipschitz ** (1 / 2))

neeta = 1.2e-3
neeta_dash = 5e-4
lipschitz_constant = 1.2
epochs = 50

mnn = MemoryNeuralNetwork(number_of_input_neurons=22, number_of_output_neurons=3, neeta=neeta, neeta_dash=neeta_dash, lipschitz_norm=lipschitz_constant, spectral_norm=True)

for epoch in range(epochs):
    error_sum = 0.0
    for i in range(len(X_train)):
        pred = mnn.feedforward(X_train[i, :])
        mnn.backprop(y_train[i, :])
        error_sum += mean_squared_error(y_train[i, :], pred.cpu().detach().numpy())
        
print(f"Epoch {epoch + 1}/{epochs}, Error: {error_sum / len(X_train):.5f}")
print(f"Epoch {epoch + 1}/{epochs}, Error: {mean_squared_error(y_train[i, :], pred.cpu().detach().numpy())}")