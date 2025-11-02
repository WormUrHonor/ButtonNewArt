import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dataset import load_fighting_game_data, normalize_input


class FightingGameNN(nn.Module):
    """
    Small neural network for predicting fighting game player actions.
    Input: 19 features (game state)
    Output: 6 binary actions (movement and attack buttons)
    """

    def __init__(self, input_size=19, hidden_sizes=[128, 64, 32], output_size=6):
        super(FightingGameNN, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers with ReLU and dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        # Output layer with sigmoid for binary classification
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        total_samples += x_batch.size(0)

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total_samples


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)

            # Calculate accuracy (threshold at 0.5)
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == y_batch).sum().item()
            total_predictions += y_batch.numel()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy


def train_model(csv_dir="data", batch_size=64, epochs=50, learning_rate=0.001,
                device=None, save_dir="models"):
    """
    Train the fighting game AI model.

    Args:
        csv_dir: Directory containing CSV replay files
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (cuda/cpu)
        save_dir: Directory to save models
    """

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Setup tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/fighting_game_{timestamp}")

    # Load data
    print("\n" + "="*50)
    train_loader, val_loader, dataset = load_fighting_game_data(
        csv_dir=csv_dir,
        batch_size=batch_size,
        num_workers=0
    )
    print("="*50 + "\n")

    # Initialize model
    model = FightingGameNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"Architecture: {model}\n")

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None

    print("Starting training...")

    try:
        pbar = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar:
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            # Validate
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)

            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Log to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)

            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_accuracy:.4f}',
                'lr': f'{current_lr:.6f}'
            })

            # Save best model state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_train_loss = train_loss
                best_val_accuracy = val_accuracy
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    finally:
        # Convert best model to ONNX
        if best_model_state is not None:
            print("\nConverting best model to ONNX...")

            # Load best model state
            model.load_state_dict(best_model_state['model_state_dict'])
            model.eval()

            # Save as ONNX
            model_path = save_path / f"best_model_{timestamp}.onnx"
            dummy_input = torch.randn(1, 19).to(device)
            torch.onnx.export(
                model,
                dummy_input,
                model_path,
                export_params=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

    print("="*50)
    if best_model_state is not None:
        print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
        print(f"  ONNX model: {model_path}")
    else:
        print("Training interrupted before any model was saved.")

    writer.close()

    return model, dataset


# def load_model(model_path, device=None):
#     """
#     Load a trained model from ONNX checkpoint.

#     Args:
#         model_path: Path to ONNX model file
#         device: Device to load model on

#     Returns:
#         model, norm_params
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Extract timestamp from model path
#     timestamp = model_path.stem.split('_')[-1]
#     model_dir = model_path.parent

#     # Load normalization parameters
#     norm_path = model_dir / f"normalization_{timestamp}.json"
#     with open(norm_path, 'r') as f:
#         norm_params = json.load(f)

#     # Load metadata
#     metadata_path = model_dir / f"metadata_{timestamp}.pkl"
#     import pickle
#     with open(metadata_path, 'rb') as f:
#         metadata = pickle.load(f)

#     # Load ONNX model into PyTorch
#     model = FightingGameNN().to(device)
#     # Note: For inference, we'll need to use ONNX Runtime or convert back
#     # For now, keeping PyTorch model structure

#     print(f"Loaded ONNX model from {model_path}")
#     print(f"Epoch: {metadata['epoch']}, Val Loss: {metadata['val_loss']:.4f}, "
#           f"Val Accuracy: {metadata['val_accuracy']:.4f}")

#     return model, norm_params


# def predict_action(model, game_state, norm_params, device=None):
#     """
#     Predict player actions given a game state.

#     Args:
#         model: Trained model
#         game_state: Array of 19 features [P1HP, P1XCoord, ...]
#         norm_params: Dictionary with min_vals and range for normalization
#         device: Device to run prediction on

#     Returns:
#         Dictionary with action probabilities
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Normalize input
#     game_state_normalized = normalize_input(game_state)
#     game_state_tensor = torch.FloatTensor([game_state_normalized]).to(device)

#     # Predict
#     model.eval()
#     with torch.no_grad():
#         output = model(game_state_tensor)
#         probabilities = output.cpu().numpy()[0]

#     actions = {
#         'Left': probabilities[0],
#         'Right': probabilities[1],
#         'Jump': probabilities[2],
#         'Dash': probabilities[3],
#         'Attack': probabilities[4],
#         'Down': probabilities[5]
#     }

#     return actions


if __name__ == "__main__":
    # Train the model
    model, dataset = train_model(
        csv_dir="data",
        batch_size=64,
        epochs=50,
        learning_rate=0.001
    )
