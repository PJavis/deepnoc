"""
Training loop for deepNoC and simplified models.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.deepnoc.architecture import DeepNoC, DeepNoCSimple
from models.deepnoc.losses import DeepNoCLoss, NoCOnlyLoss


def create_dataloaders(X_train, y_train, X_test, y_test,
                       batch_size=100, num_workers=0):
    """Create PyTorch DataLoaders from numpy arrays."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test),
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    return train_loader, test_loader


def train_deepnoc(X_train, y_train, X_test, y_test,
                   num_classes=5,
                   epochs=2000,
                   batch_size=100,
                   lr=1e-5,
                   beta1=0.5,
                   device=None,
                   save_dir="results",
                   model_type="full",
                   verbose=True):
    """
    Train deepNoC model - Đã sửa lỗi ReduceLROnPlateau
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Training on: {device}")
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size
    )
    
    # Create model
    if model_type == "full":
        model = DeepNoC(num_classes=num_classes).to(device)
        criterion = DeepNoCLoss()
    else:
        model = DeepNoCSimple(num_classes=num_classes).to(device)
        criterion = NoCOnlyLoss()
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model: {model_type}, Parameters: {n_params:,}")
        print(f"Training: {len(X_train)} profiles, Testing: {len(X_test)} profiles")
        print(f"Classes: {num_classes}, Epochs: {epochs}, LR: {lr}")
    
    # Optimizer (matching paper)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Learning rate scheduler - SỬA LỖI Ở ĐÂY
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=100,
        # verbose đã bị loại bỏ → thay bằng logger thủ công
    )
    
    # Training history
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'best_test_acc': 0.0, 'best_epoch': 0,
    }
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not verbose):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            if model_type == "full":
                outputs = model(batch_X)
                losses = criterion(outputs, {'profile_noc': batch_y})
                loss = losses['total']
                preds = outputs['profile_noc'].argmax(dim=-1) + 1
            else:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                preds = logits.argmax(dim=-1) + 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(batch_X)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # === Evaluation ===
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                if model_type == "full":
                    outputs = model(batch_X)
                    losses = criterion(outputs, {'profile_noc': batch_y})
                    loss = losses['total']
                    preds = outputs['profile_noc'].argmax(dim=-1) + 1
                else:
                    logits = model(batch_X)
                    loss = criterion(logits, batch_y)
                    preds = logits.argmax(dim=-1) + 1
                
                test_loss += loss.item() * len(batch_X)
                test_correct += (preds == batch_y).sum().item()
                test_total += len(batch_y)
        
        test_loss /= test_total
        test_acc = test_correct / test_total
        
        # Update scheduler
        scheduler.step(test_acc)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            history['best_test_acc'] = best_acc
            history['best_epoch'] = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'num_classes': num_classes,
                'model_type': model_type,
            }, os.path.join(save_dir, f'best_model_{model_type}.pt'))
        
        # Print progress
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | "
                  f"Best Test Acc: {best_acc:.3f} (ep {history['best_epoch']})")
        
        # Early save every 200 epochs
        if (epoch + 1) % 200 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(save_dir, f'checkpoint_{model_type}_ep{epoch+1}.pt'))
    
    # Save final history
    with open(os.path.join(save_dir, f'history_{model_type}.json'), 'w') as f:
        json.dump({k: [float(x) if isinstance(x, (int,float)) else x 
                      for x in v] if isinstance(v, list) else v 
                   for k, v in history.items()}, f, indent=2)
    
    if verbose:
        print(f"\nTraining completed! Best test accuracy: {best_acc:.4f} at epoch {history['best_epoch']}")
    
    return model, history