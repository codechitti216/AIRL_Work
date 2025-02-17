import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class FANLayer(nn.Module):
    def __init__(self, in_features, d_p, d_p_bar, activation):
        super(FANLayer, self).__init__()
        # Initialize weights with Xavier initialization for better convergence
        self.Wp = nn.Parameter(torch.randn(in_features, d_p))  # Fourier component
        self.Wp_bar = nn.Parameter(torch.randn(in_features, d_p_bar))  # Non-periodic component
        self.Bp_bar = nn.Parameter(torch.zeros(d_p_bar))  # Bias for non-periodic term
        
        # Linear transformations
        self.Win = nn.Linear(in_features, in_features, bias=False)  # Scaling input for frequency components
        self.projection = nn.Linear(in_features, 2 * d_p + d_p_bar, bias=False)  # Output projection
        self.activation = activation  # Activation function (e.g., GELU, ReLU)

        # Initialize projections with Xavier
        init.xavier_uniform_(self.Win.weight)
        init.xavier_uniform_(self.projection.weight)

    def forward(self, x):
        # Ensure input is of shape [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Compute frequency-scaled input, periodic (cosine and sine) and non-periodic term
        frequency_scaled_x = self.Win(x)
        cos_term = torch.cos(torch.matmul(frequency_scaled_x, self.Wp))  # Cosine Fourier component
        sin_term = torch.sin(torch.matmul(frequency_scaled_x, self.Wp))  # Sine Fourier component
        non_periodic_term = self.activation(torch.matmul(x, self.Wp_bar) + self.Bp_bar)  # Non-periodic term
        
        # Combine periodic and non-periodic terms
        output = torch.cat([cos_term, sin_term, non_periodic_term], dim=-1)
        
        # Linear projection to ensure dimensionality matches
        x_proj = self.projection(x)
        
        return output + x_proj  # Return the final output after combining

class FAN(nn.Module):
    def __init__(self, number_of_input_neurons=22, number_of_hidden_neurons=60, number_of_output_neurons=3, 
                 num_layers=3, activation=nn.GELU(), seed_value=16981, learning_rate=1e-3, 
                 lambda_reg=0.001, weight_decay=1e-4):
        super(FAN, self).__init__()
        
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.num_layers = num_layers
        self.activation = activation
        self.weight_decay = weight_decay
        
        # Prepare layers
        self.layers = nn.ModuleList()
        d_p = number_of_hidden_neurons // 4  # Set size of Fourier components
        d_p_bar = number_of_hidden_neurons  # Non-periodic components
        
        for _ in range(num_layers - 1):  # Build the network layers
            self.layers.append(FANLayer(number_of_input_neurons, d_p, d_p_bar, activation))
            number_of_input_neurons = 2 * d_p + d_p_bar  # Update input size for next layer
        
        # Final weight matrix and bias for the output layer
        self.WL = nn.Parameter(torch.randn(number_of_input_neurons, number_of_output_neurons))
        self.BL = nn.Parameter(torch.zeros(number_of_output_neurons))
        
        # Optimizer and loss function
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_function = nn.MSELoss()
        
        # Initialize final layer weights with Xavier
        init.xavier_uniform_(self.WL)
        
        self.prev_output = torch.zeros(number_of_output_neurons, device=self.device)
        
        self.to(self.device)

    def forward(self, input_array):
        # Ensure input is of shape [batch_size, features]
        x = torch.tensor(input_array, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension is present
        
        # Pass through each FAN layer
        for layer in self.layers:
            x = layer(x)
        
        # Final output computation
        output = torch.matmul(x, self.WL) + self.BL
        self.prev_output = output.clone()  # Store previous output for potential use
        
        return output.squeeze(0)  # Remove batch dimension for single input case

    def compute_loss(self, predictions, targets):
        # Compute MSE loss with regularization
        mse_loss = self.loss_function(predictions, targets)
        reg_loss = self.lambda_reg * torch.norm(self.WL, p=2)
        return mse_loss + reg_loss

    def backpropagate(self, input_array, target_array):
        # Perform backpropagation
        self.optimizer.zero_grad()
        predictions = self.forward(input_array)  # Correct method call
        targets = torch.tensor(target_array, dtype=torch.float32, device=self.device)
        loss = self.compute_loss(predictions, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
