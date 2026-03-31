# utils/data_loader.py
import torch
import numpy as np
from typing import Tuple

def generate_synthetic_data(num_nodes: int = 100,
                            seq_len: int = 50,
                            train_samples: int = 5000,
                            val_samples: int = 1000,
                            noise_level: float = 0.05,
                            fault_rate: float = 0.1) -> Tuple:
    """
    Generate synthetic network data for training
    
    Args:
        num_nodes: number of network nodes
        seq_len: sequence length
        train_samples: number of training samples
        val_samples: number of validation samples
        noise_level: observation noise level
        fault_rate: probability of fault at each node
        
    Returns:
        adj_matrix, X_train, y_train, X_val, y_val
    """
    
    # Generate hierarchical topology (adjacency matrix)
    adj_matrix = generate_hierarchical_topology(num_nodes)
    
    def generate_sample():
        # Generate fault pattern
        faults = np.random.binomial(1, fault_rate, size=(seq_len, num_nodes))
        
        # Propagate faults through topology
        fault_propagated = propagate_faults(faults, adj_matrix.numpy())
        
        # Generate QoS observations
        qos_data = generate_qos_observations(
            fault_propagated, 
            seq_len, 
            num_nodes,
            noise_level
        )
        
        return qos_data, fault_propagated
    
    # Generate training data
    X_train, y_train = [], []
    for _ in range(train_samples):
        X, y = generate_sample()
        X_train.append(X)
        y_train.append(y)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Generate validation data
    X_val, y_val = [], []
    for _ in range(val_samples):
        X, y = generate_sample()
        X_val.append(X)
        y_val.append(y)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    return adj_matrix, X_train, y_train, X_val, y_val


def generate_hierarchical_topology(num_nodes: int) -> torch.Tensor:
    """Generate hierarchical network topology"""
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    
    # Create layers
    num_layers = 4
    nodes_per_layer = num_nodes // num_layers
    
    for layer in range(num_layers - 1):
        start_idx = layer * nodes_per_layer
        end_idx = (layer + 1) * nodes_per_layer
        
        next_start = (layer + 1) * nodes_per_layer
        next_end = min((layer + 2) * nodes_per_layer, num_nodes)
        
        # Connect to next layer
        for i in range(start_idx, end_idx):
            num_connections = np.random.randint(1, 4)
            targets = np.random.choice(
                range(next_start, next_end), 
                size=min(num_connections, next_end - next_start),
                replace=False
            )
            for t in targets:
                adj_matrix[i, t] = 1.0
    
    return adj_matrix


def propagate_faults(faults: np.ndarray, 
                     adj_matrix: np.ndarray) -> np.ndarray:
    """Propagate faults through network topology"""
    seq_len, num_nodes = faults.shape
    propagated = faults.copy()
    
    for t in range(seq_len):
        for _ in range(3):  # Propagation steps
            new_faults = propagated[t].copy()
            for i in range(num_nodes):
                if propagated[t, i] == 1:
                    # Propagate to children
                    children = np.where(adj_matrix[i] > 0)[0]
                    for child in children:
                        if np.random.random() < 0.7:  # 70% propagation probability
                            new_faults[child] = 1
            propagated[t] = new_faults
    
    return propagated


def generate_qos_observations(faults: np.ndarray,
                               seq_len: int,
                               num_nodes: int,
                               noise_level: float) -> np.ndarray:
    """Generate QoS observations based on fault states"""
    # Base QoS values [latency, packet_loss, throughput, availability]
    base_qos = np.array([20.0, 0.5, 100.0, 99.0])
    
    # Fault impact
    fault_impact = np.array([100.0, 10.0, -80.0, -20.0])
    
    observations = np.zeros((seq_len, num_nodes, 4))
    
    for t in range(seq_len):
        for i in range(num_nodes):
            # Base QoS with noise
            qos = base_qos + np.random.normal(0, noise_level * base_qos)
            
            # Add fault impact
            if faults[t, i] == 1:
                qos += fault_impact * np.random.uniform(0.5, 1.0)
            
            # Ensure valid ranges
            qos[0] = max(1.0, qos[0])  # latency
            qos[1] = max(0.0, min(100.0, qos[1]))  # packet loss
            qos[2] = max(0.0, qos[2])  # throughput
            qos[3] = max(0.0, min(100.0, qos[3]))  # availability
            
            observations[t, i] = qos
    
    # Normalize
    observations[:, :, 0] /= 100.0  # latency
    observations[:, :, 1] /= 10.0   # packet loss
    observations[:, :, 2] /= 100.0  # throughput
    observations[:, :, 3] /= 100.0  # availability
    
    return observations
