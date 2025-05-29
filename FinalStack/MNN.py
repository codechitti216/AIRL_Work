import torch
import torch.nn as nn

class MemoryNeuralNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=15, number_of_hidden_neurons=100, number_of_output_neurons=3,
                 learning_rate=0.001, learning_rate_2=0.0005, learning_rate_3=0.00025,
                 dropout_rate=0.0, lipschitz_constant=1.0, spectral_norm=True, seed_value=16981):
        super(MemoryNeuralNetwork, self).__init__()
        torch.manual_seed(seed_value)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.N_in = number_of_input_neurons
        self.N_hidden = number_of_hidden_neurons
        self.N_out = number_of_output_neurons
        self.N_mem = 3

        # Main learning rate for neural network weights
        self.learning_rate = learning_rate
        
        # Separate learning rates for each memory neuron
        self.learning_rate_mem1 = learning_rate_2  # Fastest memory (v1)
        self.learning_rate_mem2 = learning_rate_2 * 0.5  # Medium memory (v2)
        self.learning_rate_mem3 = learning_rate_3  # Slowest memory (v3)
        
        self.lipschitz_constant = lipschitz_constant
        self.spectral_norm = spectral_norm

        # Memory mixing parameters
        self.alpha_input_layer = nn.Parameter(torch.rand(self.N_in, device=self.device))
        self.alpha_hidden_layer = nn.Parameter(torch.rand(self.N_hidden, device=self.device))
        self.alpha_last_layer = nn.Parameter(torch.rand(self.N_out, self.N_mem, device=self.device))
        
        # Memory cascade weights
        self.beta = nn.Parameter(torch.rand(self.N_out, self.N_mem, device=self.device))
        
        # Memory feedback weights
        self.memory_feedback = nn.Parameter(torch.rand(self.N_mem, self.N_mem, device=self.device))

        # Initialize parameters
        self.alpha_input_layer.data.clamp_(0.0, 1.0)
        self.alpha_hidden_layer.data.clamp_(0.0, 1.0)
        self.alpha_last_layer.data.clamp_(0.0, 1.0)
        self.beta.data.clamp_(0.0, 1.0)
        self.memory_feedback.data.clamp_(-0.1, 0.1)  # Small feedback weights

        # Neural network weights
        self.weights_input_to_hidden_nn = nn.Parameter(torch.empty(self.N_in, self.N_hidden, device=self.device))
        self.weights_input_to_hidden_mn = nn.Parameter(torch.empty(self.N_in, self.N_hidden, device=self.device))
        self.weights_hidden_to_output_nn = nn.Parameter(torch.empty(self.N_hidden, self.N_out, device=self.device))
        self.weights_hidden_to_output_mn = nn.Parameter(torch.empty(self.N_hidden, self.N_out, device=self.device))

        # Initialize weights with smaller gain for stability
        torch.nn.init.xavier_uniform_(self.weights_input_to_hidden_nn, gain=0.05)
        torch.nn.init.xavier_uniform_(self.weights_input_to_hidden_mn, gain=0.05)
        torch.nn.init.xavier_uniform_(self.weights_hidden_to_output_nn, gain=0.05)
        torch.nn.init.xavier_uniform_(self.weights_hidden_to_output_mn, gain=0.05)

        # Initialize memory states with small random values
        self.prev_output_of_input_layer_nn = torch.randn(self.N_in, device=self.device) * 0.01
        self.prev_output_of_input_layer_mn = torch.randn(self.N_in, device=self.device) * 0.01
        self.prev_output_of_hidden_layer_nn = torch.randn(self.N_hidden, device=self.device) * 0.01
        self.prev_output_of_hidden_layer_mn = torch.randn(self.N_hidden, device=self.device) * 0.01
        self.prev_output_of_nn = torch.randn(self.N_out, device=self.device) * 0.01
        
        # Initialize memory cascade with correct dimensions
        if self.N_out == 3:  # Velocity model
            self.prev_output_of_mn_cascade = torch.randn(self.N_out, self.N_mem, device=self.device) * 0.01
        else:  # Beam model
            # For beam model, initialize with 3 memory neurons but 4 outputs
            self.prev_output_of_mn_cascade = torch.randn(self.N_out, self.N_mem, device=self.device) * 0.01
            # Initialize the last output with zeros to avoid feedback issues
            self.prev_output_of_mn_cascade[3, :] = 0.0

        self.to(self.device)

    def forward(self, input_array):
        # Only print shapes at the start and end of memory cascade
        x = input_array.to(self.device)
        x = torch.clamp(x, -10.0, 10.0)

        x_mn = self.alpha_input_layer * self.prev_output_of_input_layer_nn + \
               (1.0 - self.alpha_input_layer) * self.prev_output_of_input_layer_mn
        x_mn = torch.clamp(x_mn, -10.0, 10.0)

        if x.dim() == 2 and x.shape[0] == 1:
            x_mem = x.squeeze(0)
        elif x.dim() == 1:
            x_mem = x
        else:
            raise ValueError(f"Input x must be 1D or [1, N], got shape {x.shape}")

        # For computation, ensure x is 2D
        if x_mem.dim() == 1:
            x_comp = x_mem.unsqueeze(0)
        else:
            x_comp = x_mem

        hidden_input_nn = torch.matmul(x_comp, self.weights_input_to_hidden_nn)
        hidden_input_mn = torch.matmul(x_mn, self.weights_input_to_hidden_mn)
        
        # Clamp hidden inputs
        hidden_input_nn = torch.clamp(hidden_input_nn, -10.0, 10.0)
        hidden_input_mn = torch.clamp(hidden_input_mn, -10.0, 10.0)
        
        hidden_raw = hidden_input_nn + hidden_input_mn
        hidden_output_nn = torch.tanh(hidden_raw)
        hidden_output_mn = self.alpha_hidden_layer * self.prev_output_of_hidden_layer_nn + \
                           (1.0 - self.alpha_hidden_layer) * self.prev_output_of_hidden_layer_mn
        
        # Clamp hidden outputs
        hidden_output_nn = torch.clamp(hidden_output_nn, -1.0, 1.0)
        hidden_output_mn = torch.clamp(hidden_output_mn, -1.0, 1.0)

        # Memory cascade with feedback
        v1 = self.alpha_last_layer[:, 0] * self.prev_output_of_nn + \
             (1.0 - self.alpha_last_layer[:, 0]) * self.prev_output_of_mn_cascade[:, 0]
        v1 = torch.clamp(v1, -1.0, 1.0)
        
        # Add feedback from v3 to v1, handling different output sizes
        if self.N_out == 3:  # Velocity model
            v1 = v1 + 0.1 * torch.matmul(self.prev_output_of_mn_cascade[:, 2], self.memory_feedback[0])
        else:  # Beam model
            # Properly slice v1 to get first 3 elements and ensure correct shape
            v1_slice = v1.squeeze(0)[:3]  # Remove batch dim and take first 3 elements
            feedback = torch.matmul(v1_slice, self.memory_feedback[0])
            v1 = v1 + 0.1 * feedback.unsqueeze(0)  # Add batch dim back
        
        v2 = self.alpha_last_layer[:, 1] * v1 + \
             (1.0 - self.alpha_last_layer[:, 1]) * self.prev_output_of_mn_cascade[:, 1]
        v2 = torch.clamp(v2, -1.0, 1.0)
        
        # Add feedback from v1 to v2, handling different output sizes
        if self.N_out == 3:  # Velocity model
            v2 = v2 + 0.1 * torch.matmul(v1, self.memory_feedback[1])
        else:  # Beam model
            # Properly slice v1 to get first 3 elements and ensure correct shape
            v1_slice = v1.squeeze(0)[:3]  # Remove batch dim and take first 3 elements
            feedback = torch.matmul(v1_slice, self.memory_feedback[1])
            v2 = v2 + 0.1 * feedback.unsqueeze(0)  # Add batch dim back
        
        v3 = self.alpha_last_layer[:, 2] * v2 + \
             (1.0 - self.alpha_last_layer[:, 2]) * self.prev_output_of_mn_cascade[:, 2]
        v3 = torch.clamp(v3, -1.0, 1.0)
        
        # Add feedback from v2 to v3, handling different output sizes
        if self.N_out == 3:  # Velocity model
            v3 = v3 + 0.1 * torch.matmul(v2, self.memory_feedback[2])
        else:  # Beam model
            # Properly slice v2 to get first 3 elements and ensure correct shape
            v2_slice = v2.squeeze(0)[:3]  # Remove batch dim and take first 3 elements
            feedback = torch.matmul(v2_slice, self.memory_feedback[2])
            v3 = v3 + 0.1 * feedback.unsqueeze(0)  # Add batch dim back

        memory_output = self.beta[:, 0] * v1 + self.beta[:, 1] * v2 + self.beta[:, 2] * v3
        memory_output = torch.clamp(memory_output, -1.0, 1.0)

        y_nn = torch.matmul(hidden_output_nn, self.weights_hidden_to_output_nn)
        y_mn = torch.matmul(hidden_output_mn, self.weights_hidden_to_output_mn)
        
        # Clamp neural network outputs
        y_nn = torch.clamp(y_nn, -1.0, 1.0)
        y_mn = torch.clamp(y_mn, -1.0, 1.0)
        
        y = y_nn + y_mn + memory_output
        y = torch.clamp(y, -1.0, 1.0)

        # Store previous outputs with clamping
        self.prev_output_of_input_layer_nn = torch.clamp(x.detach().mean(dim=0), -1.0, 1.0)
        self.prev_output_of_input_layer_mn = torch.clamp(x_mn.detach(), -1.0, 1.0)
        self.prev_output_of_hidden_layer_nn = torch.clamp(hidden_output_nn.detach(), -1.0, 1.0)
        self.prev_output_of_hidden_layer_mn = torch.clamp(hidden_output_mn.detach(), -1.0, 1.0)
        self.prev_output_of_nn = torch.clamp(y.detach(), -1.0, 1.0)
        
        # Store memory cascade outputs with proper dimensions
        if self.N_out == 3:  # Velocity model
            self.prev_output_of_mn_cascade = torch.clamp(torch.stack([v1, v2, v3], dim=1).detach(), -1.0, 1.0)
        else:  # Beam model
            # For beam model, store all outputs but keep last one zeroed
            cascade = torch.stack([v1, v2, v3], dim=1)
            cascade = torch.clamp(cascade.detach(), -1.0, 1.0)
            self.prev_output_of_mn_cascade = cascade

        return y

    def backprop(self, y_des):
        y_des = y_des.to(self.device).detach()
        y_des = torch.clamp(y_des, -1.0, 1.0)  # Clamp target values
        
        error = self.prev_output_of_nn - y_des
        error = torch.clamp(error, -1.0, 1.0)  # Clamp error

        with torch.no_grad():
            # Reduce learning rates for stability
            lr = self.learning_rate * 0.1
            lr_mem1 = self.learning_rate_mem1 * 0.1  # Fastest memory
            lr_mem2 = self.learning_rate_mem2 * 0.1  # Medium memory
            lr_mem3 = self.learning_rate_mem3 * 0.1  # Slowest memory
            
            grad_hidden_to_output_nn = torch.matmul(self.prev_output_of_hidden_layer_nn.unsqueeze(1), error.unsqueeze(0))
            grad_input_to_hidden_nn = torch.matmul(self.prev_output_of_input_layer_nn.unsqueeze(1), error.unsqueeze(0))
            grad_hidden_to_output_mn = torch.matmul(self.prev_output_of_hidden_layer_mn.unsqueeze(1), error.unsqueeze(0))
            grad_input_to_hidden_mn = torch.matmul(self.prev_output_of_input_layer_mn.unsqueeze(1), error.unsqueeze(0))

            # Clamp gradients
            for grad in [grad_hidden_to_output_nn, grad_input_to_hidden_nn, 
                        grad_hidden_to_output_mn, grad_input_to_hidden_mn]:
                grad.clamp_(-0.1, 0.1)

            self.weights_hidden_to_output_nn -= lr * grad_hidden_to_output_nn
            self.weights_input_to_hidden_nn -= lr * grad_input_to_hidden_nn
            self.weights_hidden_to_output_mn -= lr * grad_hidden_to_output_mn
            self.weights_input_to_hidden_mn -= lr * grad_input_to_hidden_mn

            # Update beta with memory-specific learning rates
            grad_beta = error.unsqueeze(1) * self.prev_output_of_mn_cascade
            grad_beta = torch.clamp(grad_beta, -0.1, 0.1)
            
            # Apply different learning rates to each memory neuron's beta
            self.beta[:, 0] -= lr_mem1 * grad_beta[:, 0]  # Fastest memory
            self.beta[:, 1] -= lr_mem2 * grad_beta[:, 1]  # Medium memory
            self.beta[:, 2] -= lr_mem3 * grad_beta[:, 2]  # Slowest memory

            # Update memory feedback weights
            grad_feedback = error.unsqueeze(1) * self.memory_feedback
            grad_feedback = torch.clamp(grad_feedback, -0.1, 0.1)
            self.memory_feedback -= lr * grad_feedback

            v1_prev = self.prev_output_of_mn_cascade[:, 0]
            v2_prev = self.prev_output_of_mn_cascade[:, 1]
            v3_prev = self.prev_output_of_mn_cascade[:, 2]
            y_prev = self.prev_output_of_nn

            alpha1 = self.alpha_last_layer[:, 0]
            alpha2 = self.alpha_last_layer[:, 1]
            alpha3 = self.alpha_last_layer[:, 2]

            dv1_dalpha1 = y_prev - v1_prev
            dv2_dalpha2 = v1 - v2_prev
            dv3_dalpha3 = v2 - v3_prev

            grad_alpha1 = error * self.beta[:, 0] * dv1_dalpha1
            grad_alpha2 = error * self.beta[:, 1] * dv2_dalpha2
            grad_alpha3 = error * self.beta[:, 2] * dv3_dalpha3

            # Clamp alpha gradients
            grad_alpha1 = torch.clamp(grad_alpha1, -0.1, 0.1)
            grad_alpha2 = torch.clamp(grad_alpha2, -0.1, 0.1)
            grad_alpha3 = torch.clamp(grad_alpha3, -0.1, 0.1)

            # Update alphas with memory-specific learning rates
            self.alpha_last_layer[:, 0] -= lr_mem1 * grad_alpha1  # Fastest memory
            self.alpha_last_layer[:, 1] -= lr_mem2 * grad_alpha2  # Medium memory
            self.alpha_last_layer[:, 2] -= lr_mem3 * grad_alpha3  # Slowest memory

            # Clamp and normalize parameters
            self.alpha_input_layer.data = torch.clamp(self.alpha_input_layer, 0.0, 1.0)
            self.alpha_hidden_layer.data = torch.clamp(self.alpha_hidden_layer, 0.0, 1.0)
            self.alpha_last_layer.data = torch.clamp(self.alpha_last_layer, 0.0, 1.0)
            self.beta.data = torch.clamp(self.beta, 0.0, 1.0)
            
            # Normalize beta with larger epsilon
            beta_norm = torch.norm(self.beta, dim=1, keepdim=True) + 1e-4
            self.beta.data = self.beta / beta_norm

            if self.spectral_norm:
                epsilon = 1e-4  # Increased epsilon
                for w in [self.weights_input_to_hidden_nn, self.weights_input_to_hidden_mn,
                          self.weights_hidden_to_output_nn, self.weights_hidden_to_output_mn]:
                    w_norm = torch.norm(w, p=2) + epsilon
                    w.data = (w / w_norm) * (self.lipschitz_constant ** 0.5)
                beta_norm = torch.norm(self.beta, p=2) + epsilon
                self.beta.data = (self.beta / beta_norm) * (self.lipschitz_constant ** 0.5)

        return error
