import torch
import torch.nn as nn
import torch.nn.functional as F

class FANLayer(nn.Module):
    def __init__(self, in_features, d_p, d_p_bar, activation=nn.GELU()):
        super(FANLayer, self).__init__()
        self.Wp = nn.Parameter(torch.randn(in_features, d_p))
        self.Wp_bar = nn.Parameter(torch.randn(in_features, d_p_bar))
        self.Bp_bar = nn.Parameter(torch.zeros(d_p_bar))
        self.Win = nn.Linear(in_features, in_features, bias=False)  
        
        self.projection = nn.Linear(in_features, 2 * d_p + d_p_bar, bias=False)  
        self.activation = activation  

    def forward(self, x):
        if x.dim() == 1:  
            x = x.unsqueeze(0)  
        
        frequency_scaled_x = self.Win(x)  
        cos_term = torch.cos(torch.matmul(frequency_scaled_x, self.Wp))
        sin_term = torch.sin(torch.matmul(frequency_scaled_x, self.Wp))
        non_periodic_term = self.activation(torch.matmul(x, self.Wp_bar) + self.Bp_bar)

        output = torch.cat([cos_term, sin_term, non_periodic_term], dim=-1)

        
        x_proj = self.projection(x)  
        
        return output + x_proj  


class FAN(nn.Module):
    def __init__(self, number_of_input_neurons=22, number_of_hidden_neurons=60, number_of_output_neurons=3, 
                 num_layers=3, activation=nn.GELU(), seed_value=16981, learning_rate=1e-3, lambda_reg=0.001):
        super(FAN, self).__init__()
        
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg  
        
        self.layers = nn.ModuleList()
        d_p = number_of_hidden_neurons // 4
        d_p_bar = number_of_hidden_neurons
        
        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(number_of_input_neurons, d_p, d_p_bar, activation))
            number_of_input_neurons = 2 * d_p + d_p_bar  
        
        self.WL = nn.Parameter(torch.randn(number_of_input_neurons, number_of_output_neurons))
        self.BL = nn.Parameter(torch.zeros(number_of_output_neurons))
        
        self.prev_output = torch.zeros(number_of_output_neurons, device=self.device)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.loss_function = nn.MSELoss()
        
        self.to(self.device)

    def feedforward(self, input_array):
        x = torch.tensor(input_array, dtype=torch.float32, device=self.device)
        if x.dim() == 1:  
            x = x.unsqueeze(0)  

        for layer in self.layers:
            x = layer(x)

        output = torch.matmul(x, self.WL) + self.BL  
        self.prev_output = output.clone()

        return output.squeeze(0)  

    def compute_loss(self, predictions, targets):
        mse_loss = self.loss_function(predictions, targets)
        reg_loss = self.lambda_reg * torch.norm(self.WL, p=2)  
        return mse_loss + reg_loss  

    def backpropagate(self, input_array, target_array):
        self.optimizer.zero_grad()
        predictions = self.feedforward(input_array)
        targets = torch.tensor(target_array, dtype=torch.float32, device=self.device)
        loss = self.compute_loss(predictions, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
