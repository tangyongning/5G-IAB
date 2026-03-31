# models/temporal_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GraphGRUCell(nn.Module):
    """
    Graph-based GRU Cell for Temporal Graph Learning (Section III-E)
    
    Implements Eq. (12):
    h_v(t) = ψ(h_v(t-1), y_v(t), Σ_{u∈P(v)} w_{u,v}·h_u(t-1))
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_nodes: int,
                 adjacency_matrix: torch.Tensor):
        super(GraphGRUCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Register adjacency matrix
        self.register_buffer('adjacency_matrix', adjacency_matrix)
        
        # GRU gates
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Graph convolution for neighbor aggregation
        self.neighbor_transform = nn.Linear(hidden_dim, hidden_dim)
        
    def aggregate_neighbors(self,
                            hidden_states: torch.Tensor,
                            adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Aggregate hidden states from upstream neighbors
        
        Args:
            hidden_states: (batch_size, num_nodes, hidden_dim)
            adjacency_matrix: (num_nodes, num_nodes)
            
        Returns:
            aggregated: (batch_size, num_nodes, hidden_dim)
        """
        # Transform neighbor states
        transformed = self.neighbor_transform(hidden_states)  # (B, N, H)
        
        # Aggregate using adjacency matrix (only upstream)
        # Σ_{u∈P(v)} w_{u,v}·h_u(t-1)
        aggregated = torch.matmul(adjacency_matrix.T, transformed)  # (B, N, H)
        
        # Normalize by degree
        degree = adjacency_matrix.sum(dim=0, keepdim=True) + 1e-8
        aggregated = aggregated / degree.T
        
        return aggregated
    
    def forward(self,
                input_features: torch.Tensor,
                hidden_states: torch.Tensor,
                adjacency_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Graph GRU update (Eq. 12)
        
        Args:
            input_features: (batch_size, num_nodes, input_dim) - QoS observations
            hidden_states: (batch_size, num_nodes, hidden_dim) - h_v(t-1)
            adjacency_matrix: (num_nodes, num_nodes) - optional override
            
        Returns:
            new_hidden: (batch_size, num_nodes, hidden_dim) - h_v(t)
        """
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix
        
        batch_size = input_features.size(0)
        
        # Aggregate neighbor hidden states
        neighbor_agg = self.aggregate_neighbors(hidden_states, adjacency_matrix)
        
        # Concatenate input with neighbor aggregation
        combined = torch.cat([input_features, neighbor_agg], dim=-1)  # (B, N, 2H)
        
        # Compute GRU gates
        z = torch.sigmoid(self.update_gate(combined))  # Update gate
        r = torch.sigmoid(self.reset_gate(combined))   # Reset gate
        
        # Candidate hidden state
        reset_hidden = r * hidden_states
        combined_candidate = torch.cat([input_features, reset_hidden], dim=-1)
        h_tilde = torch.tanh(self.candidate(combined_candidate))
        
        # Final hidden state
        new_hidden = (1 - z) * hidden_states + z * h_tilde
        
        return new_hidden


class TemporalGraphLearning(nn.Module):
    """
    Temporal Graph Learning Module (Section III-E)
    
    Implements Eq. (12)-(13):
    - Hidden state update: h_v(t) = ψ(...)
    - Fault probability: p_v(t) = σ(W_o · h_v(t) + b_o)
    """
    
    def __init__(self,
                 num_nodes: int,
                 adjacency_matrix: torch.Tensor,
                 input_dim: int = 4,  # 4 QoS metrics
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super(TemporalGraphLearning, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph GRU layers
        self.gru_cells = nn.ModuleList([
            GraphGRUCell(
                input_dim=hidden_dim if i > 0 else input_dim,
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                adjacency_matrix=adjacency_matrix
            ) for i in range(num_layers)
        ])
        
        # Output layer (Eq. 13)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize hidden states
        self.register_buffer('initial_hidden', 
                             torch.zeros(1, num_nodes, hidden_dim))
        
    def initialize_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden states for new sequence"""
        return self.initial_hidden.expand(batch_size, -1, -1).clone()
    
    def forward(self,
                observations: torch.Tensor,
                initial_hidden: Optional[torch.Tensor] = None,
                return_hidden: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through temporal graph
        
        Args:
            observations: (batch_size, seq_len, num_nodes, input_dim)
            initial_hidden: (batch_size, num_nodes, hidden_dim)
            return_hidden: whether to return final hidden states
            
        Returns:
            fault_probs: (batch_size, seq_len, num_nodes, 1)
            hidden_states: (batch_size, num_nodes, hidden_dim)
        """
        batch_size, seq_len, _, _ = observations.size()
        
        # Initialize hidden states
        if initial_hidden is None:
            hidden = self.initialize_hidden(batch_size)
        else:
            hidden = initial_hidden
        
        # Encode input
        encoded_input = self.input_encoder(observations)  # (B, T, N, H)
        
        # Process sequence
        fault_probs_list = []
        
        for t in range(seq_len):
            input_t = encoded_input[:, t, :, :]  # (B, N, H)
            
            # Pass through GRU layers
            for i, gru_cell in enumerate(self.gru_cells):
                if i == 0:
                    hidden = gru_cell(input_t, hidden)
                else:
                    hidden = gru_cell(hidden, hidden)
            
            # Compute fault probability (Eq. 13)
            # p_v(t) = σ(W_o · h_v(t) + b_o)
            fault_prob = self.output_layer(hidden)  # (B, N, 1)
            fault_probs_list.append(fault_prob)
        
        # Stack probabilities
        fault_probs = torch.stack(fault_probs_list, dim=1)  # (B, T, N, 1)
        
        if return_hidden:
            return fault_probs, hidden
        
        return fault_probs
