# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.tcca_framework import TCCAFramework
from utils.data_loader import generate_synthetic_data

def train_tcca(num_nodes=100,
               seq_len=50,
               batch_size=32,
               epochs=100,
               lr=0.001,
               device='cuda'):
    """
    Train TCCA framework
    
    Args:
        num_nodes: number of network nodes
        seq_len: sequence length
        batch_size: training batch size
        epochs: number of training epochs
        lr: learning rate
        device: training device
    """
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    adj_matrix, X_train, y_train, X_val, y_val = generate_synthetic_data(
        num_nodes=num_nodes,
        seq_len=seq_len,
        train_samples=5000,
        val_samples=1000
    )
    
    # Move to device
    adj_matrix = adj_matrix.to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TCCAFramework(
        num_nodes=num_nodes,
        adjacency_matrix=adj_matrix,
        input_dim=4,
        hidden_dim=64,
        lambda_reliability=0.1
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch, y_batch)
            loss = output['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                output = model(X_batch, y_batch)
                val_loss += output['loss'].item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_tcca_model.pth')
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = train_tcca(
        num_nodes=100,
        seq_len=50,
        batch_size=32,
        epochs=100,
        lr=0.001,
        device=device
    )
