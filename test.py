import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
from dataset import load_fighting_game_data


def get_most_recent_model(model_dir="models"):
    """Find the most recently created ONNX model."""
    model_path = Path(model_dir)
    onnx_files = sorted(model_path.glob("best_model_*.onnx"), key=lambda p: p.stat().st_mtime)

    if not onnx_files:
        raise FileNotFoundError(f"No ONNX models found in {model_dir}")

    return onnx_files[-1]


def load_onnx_model(model_path):
    """Load ONNX model."""
    # Extract timestamp from model path
    timestamp = model_path.stem.split('_')[-1]
    model_dir = model_path.parent

    # Create ONNX Runtime session
    session = ort.InferenceSession(str(model_path))

    print(f"Loaded model: {model_path.name}")

    return session


def predict_onnx(session, game_state):
    """Run inference using ONNX model."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    result = session.run([output_name], {input_name: game_state.astype(np.float32)})
    return result[0]


def evaluate_match(csv_dir="data", model_dir="models"):
    """Evaluate model on a full match."""

    # Load most recent model
    model_path = get_most_recent_model(model_dir)
    session = load_onnx_model(model_path)

    # Load dataset (just to get a match, no training)
    print("Loading dataset...")
    _, _, dataset = load_fighting_game_data(csv_dir=csv_dir, batch_size=1)

    print("\n" + "="*70)
    print("Evaluating on full match:")
    print("="*70)

    # Get first match
    match_x, match_y = dataset.matches[0]
    action_names = ['Left', 'Right', 'Jump', 'Dash', 'Attack', 'Down']

    print(f"Match has {len(match_x)} frames\n")

    # Convert to numpy (already normalized by dataset)
    match_x_np = match_x.numpy()
    match_y_np = match_y.numpy()

    # Run predictions
    predictions = predict_onnx(session, match_x_np)

    # Threshold predictions
    predicted_actions = (predictions > 0.5).astype(int)

    # Calculate statistics
    total_frames_with_actions = 0
    correct_frames = 0

    # Print frames with actions
    for frame_idx in range(len(match_x)):
        actual = match_y_np[frame_idx]
        predicted = predicted_actions[frame_idx]

        # Get non-zero actions
        actual_nonzero = [action_names[i] for i in range(6) if actual[i] > 0.5]
        predicted_nonzero = [action_names[i] for i in range(6) if predicted[i] > 0.5]

        # Only print if there are any actions
        if actual_nonzero or predicted_nonzero:
            total_frames_with_actions += 1
            actual_str = ", ".join(actual_nonzero) if actual_nonzero else "none"
            predicted_str = ", ".join(predicted_nonzero) if predicted_nonzero else "none"

            is_correct = actual_nonzero == predicted_nonzero
            if is_correct:
                correct_frames += 1

            match_symbol = "✓" if is_correct else "✗"

            print(f"Frame {frame_idx:4d} {match_symbol} | "
                  f"Actual: {actual_str:30s} | Predicted: {predicted_str}")

    # Print summary
    print("\n" + "="*70)
    print(f"Summary:")
    print(f"  Frames with actions: {total_frames_with_actions}/{len(match_x)}")
    print(f"  Correct predictions: {correct_frames}/{total_frames_with_actions} "
          f"({100*correct_frames/total_frames_with_actions:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    evaluate_match(csv_dir="data", model_dir="models")
