# models/tcca_framework.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .causal_inference import TopologyConstrainedCausalInference
from .trust_metric import ReliabilityTrustMetric
from .temporal_graph import TemporalGraphLearning

class TCCAFramework(nn.Module):
    """
    Complete TCCA Framework (Section III)
    
    Integrates:
    - Topology-constrained causal inference
    - Reliability-oriented trust metric
    - Temporal graph learning
    
    Optimization objective (Eq. 14):
    J = L_data + λ · L_reliability
    """
    
    def __init__(self,
                 num_nodes: int,
                 adjacency_matrix: torch.Tensor,
                 input_dim: int = 4,
                 hidden_dim: int = 64,
                 lambda_reliability: float = 0.1,
                 dropout: float = 0.3):
        super(TCCAFramework, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.lambda_reliability = lambda_reliability
        
        # Core components
        self.causal_inference = TopologyConstrainedCausalInference(
            num_nodes=num_nodes,
            adjacency_matrix=adjacency_matrix,
            hidden_dim=hidden_dim
        )
        
        self.trust_metric = ReliabilityTrustMetric(
            num_nodes=num_nodes,
            adjacency_matrix=adjacency_matrix
        )
        
        self.temporal_learning = TemporalGraphLearning(
            num_nodes=num_nodes,
            adjacency_matrix=adjacency_matrix,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                observations: torch.Tensor,
                fault_labels: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TCCA framework
        
        Args:
            observations: (batch_size, seq_len, num_nodes, 4) - QoS metrics
            fault_labels: (batch_size, seq_len, num_nodes, 1) - ground truth
            return_components: whether to return intermediate components
            
        Returns:
            output_dict with:
            - fault_probs: (batch_size, seq_len, num_nodes, 1)
            - trust_scores: (batch_size, seq_len, num_nodes, 1)
            - loss: scalar (if labels provided)
            - components: dict (if return_components=True)
        """
        batch_size, seq_len, _, _ = observations.size()
        
        # Temporal graph learning
        temporal_probs, final_hidden = self.temporal_learning(
            observations, return_hidden=True
        )  # (B, T, N, 1)
        
        # Causal inference (using last time step)
        last_obs = observations[:, -1, :, :]  # (B, N, 4)
        causal_probs, upstream_influence = self.causal_inference(
            temporal_probs[:, -1, :, :], last_obs
        )  # (B, N, 1)
        
        # Trust metric
        fault_history = temporal_probs[:, :, :, 0]  # (B, T, N)
        trust_scores, trust_components, uncertainty = self.trust_metric(
            causal_probs, last_obs, fault_history
        )  # (B, N, 1)
        
        # Fuse temporal and causal predictions
        combined = torch.cat([temporal_probs[:, -1, :, :], trust_scores], dim=-1)
        final_probs = self.fusion_layer(combined)  # (B, N, 1)
        
        # Expand to sequence length
        final_probs_seq = final_probs.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        output = {
            'fault_probs': final_probs_seq,
            'temporal_probs': temporal_probs,
            'causal_probs': causal_probs.unsqueeze(1),
            'trust_scores': trust_scores.unsqueeze(1),
            'uncertainty': uncertainty.unsqueeze(1) if uncertainty is not None else None,
            'upstream_influence': upstream_influence.unsqueeze(1)
        }
        
        # Compute loss if labels provided
        if fault_labels is not None:
            loss = self.compute_loss(output, fault_labels)
            output['loss'] = loss
        
        if return_components:
            output['components'] = trust_components
        
        return output
    
    def compute_loss(self,
                     output: Dict[str, torch.Tensor],
                     labels: torch.Tensor) -> torch.Tensor:
        """
        Compute optimization objective (Eq. 14)
        
        J = L_data + λ · L_reliability
        
        Args:
            output: model output dict
            labels: (batch_size, seq_len, num_nodes, 1)
            
        Returns:
            total_loss: scalar
        """
        # Data loss: binary cross-entropy
        fault_probs = output['fault_probs']
        L_data = F.binary_cross_entropy(fault_probs, labels)
        
        # Reliability loss: topology consistency + uncertainty regularization
        trust_scores = output['trust_scores']
        uncertainty = output['uncertainty']
        
        # Encourage high trust for correct predictions
        L_trust = F.mse_loss(trust_scores, labels)
        
        # Penalize high uncertainty
        if uncertainty is not None:
            L_uncertainty = uncertainty.mean()
        else:
            L_uncertainty = 0
        
        # Topology constraint loss
        L_topo = self.causal_inference.influence_weights.pow(2).mean()
        
        # Total loss (Eq. 14)
        L_reliability = L_trust + 0.5 * L_uncertainty + 0.1 * L_topo
        total_loss = L_data + self.lambda_reliability * L_reliability
        
        return total_loss
    
    def localize_faults(self,
                        observations: torch.Tensor,
                        threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform fault localization
        
        Args:
            observations: (batch_size, seq_len, num_nodes, 4)
            threshold: fault probability threshold
            
        Returns:
            fault_predictions: (batch_size, num_nodes) - binary predictions
            confidence_scores: (batch_size, num_nodes) - trust scores
        """
        self.eval()
        
        with torch.no_grad():
            output = self.forward(observations)
            
            fault_probs = output['fault_probs'][:, -1, :, 0]  # (B, N)
            trust_scores = output['trust_scores'][:, -1, :, 0]  # (B, N)
            
            # Binary predictions
            fault_predictions = (fault_probs > threshold).float()
            
            # Confidence = probability × trust
            confidence_scores = fault_probs * trust_scores
        
        return fault_predictions, confidence_scores
