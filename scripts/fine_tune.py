#!/usr/bin/env python
import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
from datetime import datetime

# Import platform components
from data.dataset import create_data_loaders
from models.fusion.fusion_model import MultimodalFusionModel
from models.text.roberta_model import RoBERTaWrapper
from models.visual.clip_model import CLIPWrapper
from models.optimization.quantization import ModelQuantizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fine_tune')

def create_lr_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler."""
    if config['lr_scheduler'] == 'linear':
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=num_training_steps
        )
    elif config['lr_scheduler'] == 'cosine':
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=num_training_steps
        )
    elif config['lr_scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 5),
            gamma=config.get('gamma', 0.5)
        )
    else:
        return None

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, config):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        visual_features = batch['visual_features'].to(device)
        text_features = batch['text_features'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(visual_features, text_features)
        
        # Get predicted engagement
        if model.engagement_type == "regression":
            predictions = outputs["engagement"]["score"].squeeze(-1)
        else:  # classification
            predictions = torch.argmax(outputs["engagement"]["probabilities"], dim=1).float() / model.num_engagement_classes
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping if enabled
        if config.get('gradient_clipping'):
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping'])
            
        optimizer.step()
        
        # Update scheduler if step-based
        if scheduler is not None and config['lr_scheduler'] not in ['linear', 'cosine']:
            scheduler.step()
            
        # Record loss
        running_loss += loss.item() * visual_features.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log every n steps
        if batch_idx % config.get('log_every_n_steps', 10) == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Update scheduler if epoch-based
    if scheduler is not None and config['lr_scheduler'] in ['linear', 'cosine']:
        scheduler.step()
    
    # Calculate epoch loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss

def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Get data
            visual_features = batch['visual_features'].to(device)
            text_features = batch['text_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(visual_features, text_features)
            
            # Get predicted engagement
            if model.engagement_type == "regression":
                predictions = outputs["engagement"]["score"].squeeze(-1)
            else:  # classification
                predictions = torch.argmax(outputs["engagement"]["probabilities"], dim=1).float() / model.num_engagement_classes
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Record loss
            running_loss += loss.item() * visual_features.size(0)
            
            # Record predictions and labels for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate validation loss
    val_loss = running_loss / len(dataloader.dataset)
    
    # Calculate additional metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    return val_loss, mse, r2, all_predictions, all_labels

def load_and_init_model(config, device):
    """Load and initialize the model for fine-tuning."""
    base_model_path = config['model']['base_model_path']
    
    # Check if base model exists
    if os.path.exists(base_model_path):
        logger.info(f"Loading base model from {base_model_path}")
        model = MultimodalFusionModel.load(base_model_path, device=device)
    else:
        logger.info(f"Base model not found at {base_model_path}, initializing new model")
        # Initialize a new model with config
        model = MultimodalFusionModel(
            visual_dim=config['model']['visual_dim'],
            text_dim=config['model']['text_dim'],
            fusion_dim=config['model']['fusion_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout'],
            engagement_type=config['model']['engagement_type'],
            device=device
        )
    
    return model

def main():
    """Main function for fine-tuning the fusion model."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fine-tune multimodal fusion model')
    parser.add_argument('--config', type=str, default='configs/fine_tuning.yaml', help='Path to config file')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create log and checkpoint directories
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"finetune_{timestamp}"
    run_dir = checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(log_dir / f"{run_name}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Save config for this run
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize feature extraction models
    text_model = RoBERTaWrapper(
        model_name=config['features']['text_model'],
        device=device
    )
    
    visual_model = CLIPWrapper(
        model_name=config['features']['visual_model'],
        device=device
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloaders = create_data_loaders(
        config, 
        text_model=text_model, 
        visual_model=visual_model,
        device=device
    )
    
    # Load or initialize model
    logger.info("Initializing model...")
    model = load_and_init_model(config, device)
    model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Use MSE for regression
    criterion = nn.MSELoss()
    
    # Create learning rate scheduler
    num_training_steps = len(dataloaders['train']) * config['training']['num_epochs']
    scheduler = create_lr_scheduler(optimizer, config['training'], num_training_steps)
    
    # Initialize early stopping
    early_stopping = config['training'].get('early_stopping', {})
    patience = early_stopping.get('patience', 5)
    min_delta = early_stopping.get('min_delta', 0.001)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Track metrics
    train_losses = []
    val_losses = []
    val_mses = []
    val_r2s = []
    
    # Training loop
    logger.info(f"Starting training for {config['training']['num_epochs']} epochs...")
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=dataloaders['train'],
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            config=config['training']
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_mse, val_r2, _, _ = validate(
            model=model,
            dataloader=dataloaders['val'],
            criterion=criterion,
            device=device
        )
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_r2s.append(val_r2)
        
        # Log results
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        
        # Save checkpoint
        checkpoint_path = run_dir / f"epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mse': val_mse,
            'val_r2': val_r2
        }, checkpoint_path)
        
        # Check for best model
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(f"  New best model with val_loss: {val_loss:.4f}")
            
            # Save best model
            if config['logging'].get('save_best_only', True):
                best_model_path = run_dir / "best_model.pt"
                model.save(best_model_path)
                logger.info(f"  Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_mse, test_r2, test_preds, test_labels = validate(
        model=model,
        dataloader=dataloaders['test'],
        criterion=criterion,
        device=device
    )
    
    logger.info(f"Test Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  MSE: {test_mse:.4f}")
    logger.info(f"  R²: {test_r2:.4f}")
    
    # Save final model
    final_model_path = run_dir / "final_model.pt"
    model.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Optimize model
    logger.info("Optimizing model for deployment...")
    quantizer = ModelQuantizer(model)
    quantized_model = quantizer.dynamic_quantization(dtype="int8")
    
    # Save quantized model
    quant_model_path = run_dir / "quantized_model.pt"
    torch.save(quantized_model, quant_model_path)
    logger.info(f"Saved quantized model to {quant_model_path}")
    
    # Save metrics history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, val_mses, 'g-', label='Validation MSE')
    plt.plot(epochs, val_r2s, 'y-', label='Validation R²')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(run_dir / "training_metrics.png")
    
    logger.info(f"Fine-tuning complete! Results saved to {run_dir}")

if __name__ == "__main__":
    main() 