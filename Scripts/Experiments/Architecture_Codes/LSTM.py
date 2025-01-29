import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=60, number_of_output_neurons=3, neeta=1.2e-3, neeta_dash=5e-4, seed_value=16981):
        super(LSTMNetwork, self).__init__()

        torch.manual_seed(seed_value)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons

        self.neeta = neeta
        self.neeta_dash = neeta_dash

        # LSTM Parameters
        self.W_f = nn.Parameter(torch.empty(self.number_of_input_neurons, self.number_of_hidden_neurons, device=self.device))
        self.U_f = nn.Parameter(torch.empty(self.number_of_hidden_neurons, self.number_of_hidden_neurons, device=self.device))
        self.b_f = nn.Parameter(torch.zeros(self.number_of_hidden_neurons, device=self.device))

        self.W_i = nn.Parameter(torch.empty(self.number_of_input_neurons, self.number_of_hidden_neurons, device=self.device))
        self.U_i = nn.Parameter(torch.empty(self.number_of_hidden_neurons, self.number_of_hidden_neurons, device=self.device))
        self.b_i = nn.Parameter(torch.zeros(self.number_of_hidden_neurons, device=self.device))

        self.W_c = nn.Parameter(torch.empty(self.number_of_input_neurons, self.number_of_hidden_neurons, device=self.device))
        self.U_c = nn.Parameter(torch.empty(self.number_of_hidden_neurons, self.number_of_hidden_neurons, device=self.device))
        self.b_c = nn.Parameter(torch.zeros(self.number_of_hidden_neurons, device=self.device))

        self.W_o = nn.Parameter(torch.empty(self.number_of_input_neurons, self.number_of_hidden_neurons, device=self.device))
        self.U_o = nn.Parameter(torch.empty(self.number_of_hidden_neurons, self.number_of_hidden_neurons, device=self.device))
        self.b_o = nn.Parameter(torch.zeros(self.number_of_hidden_neurons, device=self.device))

        self.W_y = nn.Parameter(torch.empty(self.number_of_hidden_neurons, self.number_of_output_neurons, device=self.device))
        self.b_y = nn.Parameter(torch.zeros(self.number_of_output_neurons, device=self.device))

        # Xavier Initialization
        torch.nn.init.xavier_uniform_(self.W_f)
        torch.nn.init.xavier_uniform_(self.U_f)
        torch.nn.init.xavier_uniform_(self.W_i)
        torch.nn.init.xavier_uniform_(self.U_i)
        torch.nn.init.xavier_uniform_(self.W_c)
        torch.nn.init.xavier_uniform_(self.U_c)
        torch.nn.init.xavier_uniform_(self.W_o)
        torch.nn.init.xavier_uniform_(self.U_o)
        torch.nn.init.xavier_uniform_(self.W_y)
        
        self.prev_h = torch.zeros(self.number_of_hidden_neurons, device=self.device)
        self.prev_c = torch.zeros(self.number_of_hidden_neurons, device=self.device)
        
        self.to(self.device)

    def activation_function(self, x):
        return 15 * torch.tanh(x / 15)

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, input_array):
        self.input_nn = torch.as_tensor(input_array, dtype=torch.float32, device=self.device).clone().detach()

        forget_gate = self.sigmoid(torch.matmul(self.input_nn, self.W_f) + torch.matmul(self.prev_h, self.U_f) + self.b_f)
        input_gate = self.sigmoid(torch.matmul(self.input_nn, self.W_i) + torch.matmul(self.prev_h, self.U_i) + self.b_i)
        cell_candidate = torch.tanh(torch.matmul(self.input_nn, self.W_c) + torch.matmul(self.prev_h, self.U_c) + self.b_c)
        output_gate = self.sigmoid(torch.matmul(self.input_nn, self.W_o) + torch.matmul(self.prev_h, self.U_o) + self.b_o)
        
        self.cell_state = forget_gate * self.prev_c + input_gate * cell_candidate
        self.hidden_state = output_gate * torch.tanh(self.cell_state)
        
        output = torch.matmul(self.hidden_state, self.W_y) + self.b_y
        
        self.prev_h = self.hidden_state.clone()
        self.prev_c = self.cell_state.clone()

        return output

    def backprop(self, y_des):
        y_des = torch.as_tensor(y_des, dtype=torch.float32, device=self.device).clone().detach()
        error = self.forward(self.input_nn) - y_des
        
        # Gradient Descent Updates
        self.W_y.data -= self.neeta * torch.outer(self.hidden_state, error)
        self.b_y.data -= self.neeta * error
        
        # Backprop through time (simplified version for single-step update)
        dh = torch.matmul(error, self.W_y.t())
        dc = dh * torch.tanh(self.cell_state) * self.sigmoid(self.b_o)
        
        d_f = dc * self.prev_c * self.sigmoid(self.b_f) * (1 - self.sigmoid(self.b_f))
        d_i = dc * torch.tanh(self.cell_state) * self.sigmoid(self.b_i) * (1 - self.sigmoid(self.b_i))
        d_c = dc * self.sigmoid(self.b_i) * (1 - torch.tanh(self.cell_state) ** 2)
        d_o = dh * torch.tanh(self.cell_state) * self.sigmoid(self.b_o) * (1 - self.sigmoid(self.b_o))
        
        self.W_f.data -= self.neeta_dash * torch.outer(self.input_nn, d_f)
        self.U_f.data -= self.neeta_dash * torch.outer(self.prev_h, d_f)
        self.b_f.data -= self.neeta_dash * d_f
        
        self.W_i.data -= self.neeta_dash * torch.outer(self.input_nn, d_i)
        self.U_i.data -= self.neeta_dash * torch.outer(self.prev_h, d_i)
        self.b_i.data -= self.neeta_dash * d_i
        
        self.W_c.data -= self.neeta_dash * torch.outer(self.input_nn, d_c)
        self.U_c.data -= self.neeta_dash * torch.outer(self.prev_h, d_c)
        self.b_c.data -= self.neeta_dash * d_c
        
        self.W_o.data -= self.neeta_dash * torch.outer(self.input_nn, d_o)
        self.U_o.data -= self.neeta_dash * torch.outer(self.prev_h, d_o)
        self.b_o.data -= self.neeta_dash * d_o
