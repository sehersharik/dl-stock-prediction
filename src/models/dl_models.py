import torch
import torch.nn as nn
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class ConfigurableMLP(nn.Module):
    """
    A regularized, compact Multi-Layer Perceptron designed for noisy financial time-series.
    Defaults to use_batch_norm=False to prevent batch statistics degradation on small financial sample sizes.
    """
    def __init__(self, input_dim: int, layer_dims: list = [32, 16], dropout: float = 0.2, use_batch_norm: bool = False):
        super(ConfigurableMLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for out_dim in layer_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
            
        # Final output logit
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ConfigurableLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) for sequential financial modeling.
    Uses compact dimensional sizing [32, 16] to prevent parameter explosion.
    """
    def __init__(self, input_dim: int, lstm_dims: list = [32, 16], dense_dims: list = [16], dropout: float = 0.2):
        super(ConfigurableLSTM, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        in_size = input_dim
        
        for hidden_size in lstm_dims:
            self.lstm_layers.append(
                nn.LSTM(input_size=in_size, hidden_size=hidden_size, batch_first=True)
            )
            in_size = hidden_size
            
        self.dropout = nn.Dropout(dropout)
        
        dense_layers = []
        for d_size in dense_dims:
            dense_layers.append(nn.Linear(in_size, d_size))
            dense_layers.append(nn.ReLU())
            if dropout > 0:
                dense_layers.append(nn.Dropout(dropout))
            in_size = d_size
            
        # Final output logit
        dense_layers.append(nn.Linear(in_size, 1))
        self.fc = nn.Sequential(*dense_layers)
        
    def forward(self, x):
        out = x
        for lstm in self.lstm_layers:
            out, (hn, cn) = lstm(out)
            out = self.dropout(out)
            
        last_time_step = out[:, -1, :]
        return self.fc(last_time_step)
