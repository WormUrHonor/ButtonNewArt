import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List


def normalize_input(data):
    """Transform data using game-specific normalization."""
    normalized = data.copy()

    # P1HP (index 0): divide by 4
    normalized[:, 0] = data[:, 0] / 4.0

    # P1XCoord (index 1): divide by 8, clamp to [-1, 1]
    normalized[:, 1] = np.clip(data[:, 1] / 8.0, -1.0, 1.0)

    # P1YCoord (index 2): divide by 5, clamp to [-1, 1]
    normalized[:, 2] = np.clip(data[:, 2] / 5.0, -1.0, 1.0)

    # P1 boolean flags (indices 3-8): no normalization
    # normalized[:, 3:9] stays the same

    # P2HP (index 9): divide by 4
    normalized[:, 9] = data[:, 9] / 4.0

    # P2XCoord (index 10): divide by 8, clamp to [-1, 1]
    normalized[:, 10] = np.clip(data[:, 10] / 8.0, -1.0, 1.0)

    # P2YCoord (index 11): divide by 5, clamp to [-1, 1]
    normalized[:, 11] = np.clip(data[:, 11] / 5.0, -1.0, 1.0)

    # P2 boolean flags (indices 12-17): no normalization
    # normalized[:, 12:18] stays the same

    # Distance (index 18): divide by 16, clamp to [-1, 1]
    normalized[:, 18] = np.clip(data[:, 18] / 16.0, -1.0, 1.0)

    return normalized


class FightingGameDataset(Dataset):
    """
    PyTorch Dataset for fighting game data.
    Stores original and mirrored matches separately.
    """

    def __init__(self, csv_files: List[str]):
        """
        Args:
            csv_files: List of paths to CSV files
            normalizer: SimpleNormalizer instance (if None and fit_normalizer=True, creates new one)
            fit_normalizer: Whether to fit normalizer on this data
        """
        self.input_features = [
            'P1HP', 'P1XCoord', 'P1YCoord',
            'P1IsGrounded', 'P1IsDashing', 'P1IsAttacking',
            'P1IsHit', 'P1IsInHitStun', 'P1IsLocked',
            'P2HP', 'P2XCoord', 'P2YCoord',
            'P2IsGrounded', 'P2IsDashing', 'P2IsAttacking',
            'P2IsHit', 'P2IsInHitStun', 'P2IsLocked',
            'Distance'
        ]

        self.output_labels = [
            'P2LeftPressed', 'P2RightPressed', 'P2JumpPressed',
            'P2DashPressed', 'P2AttackPressed', 'P2DownPressed'
        ]

        self.matches = []

        # First pass: collect all data without normalization
        all_data = []
        temp_matches = []

        for csv_file in csv_files:
            try:
                # Load original
                df_original = pd.read_csv(csv_file, sep=';')
                x_original, y_original = self._process_dataframe(df_original)
                temp_matches.append((x_original, y_original))
                all_data.append(x_original.numpy())

                # Load mirrored
                df_mirrored = pd.read_csv(csv_file, sep=';')
                df_mirrored.columns = df_mirrored.columns.str.replace('P1', 'TEMP')
                df_mirrored.columns = df_mirrored.columns.str.replace('P2', 'P1')
                df_mirrored.columns = df_mirrored.columns.str.replace('TEMP', 'P2')
                x_mirrored, y_mirrored = self._process_dataframe(df_mirrored)
                temp_matches.append((x_mirrored, y_mirrored))
                all_data.append(x_mirrored.numpy())
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        # Second pass: normalize all matches
        for x, y in temp_matches:
            x_normalized = torch.FloatTensor(normalize_input(x.numpy()))
            self.matches.append((x_normalized, y))

        print(f"{len(self.matches)} matches including mirrored")
        print(f"{sum(len(x) for x, _ in self.matches)} total frames")

    def _process_dataframe(self, df: pd.DataFrame):
        """Convert dataframe to tensors without normalization."""
        # Convert boolean strings to integers
        for col in df.columns:
            if df[col].dtype == 'object':
                # Handle various string representations
                df[col] = df[col].astype(str).str.strip().map({
                    'False': 0, 'false': 0, 'FALSE': 0, '0': 0, '0.0': 0,
                    'True': 1, 'true': 1, 'TRUE': 1, '1': 1, '1.0': 1
                })
                # Fill any unmapped values with 0
                df[col] = df[col].fillna(0)

        # Convert all columns to float
        df = df.astype(float)

        # Extract features and labels
        x = df[self.input_features].values.astype(np.float32)
        y = df[self.output_labels].values.astype(np.float32)

        return torch.FloatTensor(x), torch.FloatTensor(y)

    def __len__(self):
        """Total number of frames across all matches."""
        return sum(len(x) for x, _ in self.matches)

    def __getitem__(self, idx):
        """Get individual frame."""
        current_idx = 0
        for x, y in self.matches:
            seq_len = len(x)
            if idx < current_idx + seq_len:
                local_idx = idx - current_idx
                return x[local_idx], y[local_idx]
            current_idx += seq_len

        raise IndexError("Index out of range")


def load_fighting_game_data(csv_dir: str = "data", batch_size: int = 32, num_workers: int = 0):
    """
    Load all CSV files from a directory and create DataLoader.

    Args:
        csv_dir: Directory containing CSV files
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader

    Returns:
        train_loader, val_loader, dataset
    """

    print("Loading dataset")

    # Find all CSV files
    csv_files = sorted(Path(csv_dir).glob("*.txt")) + sorted(Path(csv_dir).glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No csv files found in {csv_dir}")

    print(f"{len(csv_files)} csv files")

    # Create dataset
    dataset = FightingGameDataset([str(f) for f in csv_files])

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"{len(train_dataset)} training frames")
    print(f"{len(val_dataset)} validation frames")

    return train_loader, val_loader, dataset


# Example usage
if __name__ == "__main__":
    # Load from data directory
    train_loader, val_loader, dataset = load_fighting_game_data(batch_size=32)

    # Test standard DataLoader access
    x_batch, y_batch = next(iter(train_loader))
    print(f"Batch shape: {x_batch.numpy().shape} -> {y_batch.numpy().shape}")

    # Access matches directly
    for i, (x, y) in enumerate(dataset.matches[:1]):
        print(f"Match {i} shape: {x.numpy().shape} -> {y.numpy().shape}")
