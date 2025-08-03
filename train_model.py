#!/usr/bin/env python3
"""
Train chess AI models from pre-generated training data.
Supports multiple generations with different reward functions.
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
from chess_ai import ChessAI, GamePosition


def train_model(
    data_file: str,
    generation: int = 1,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = None,
    output_name: str = None,
) -> str:
    """Train a model from saved training data."""

    # Load training data
    print(f"📖 Loading training data from {data_file}...")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    training_positions = data["positions"]
    print(f"✅ Loaded {len(training_positions)} training positions")
    print(f"🎮 From {data['num_games']} games")

    # Create model name if not provided
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"gen{generation}_trained_{timestamp}"

    # Create AI with specified generation
    print(f"🤖 Creating Generation {generation} model...")
    ai = ChessAI(generation=generation, device=device, run_name=output_name)

    # Train the model
    print(f"🚀 Training for {epochs} epochs with batch size {batch_size}...")
    ai.train_on_data(
        training_data=training_positions, epochs=epochs, batch_size=batch_size
    )

    # Save the trained model
    model_path = f"{ai.run_dir}/final_model.pth"
    ai.save_model(model_path)

    print(f"💾 Model saved to {model_path}")

    # Save training metadata
    metadata = {
        "generation": generation,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(ai.device),
        "training_data_file": data_file,
        "num_training_positions": len(training_positions),
        "num_source_games": data["num_games"],
        "trained_timestamp": datetime.now().isoformat(),
    }

    metadata_path = f"{ai.run_dir}/training_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"📋 Training metadata saved to {metadata_path}")

    return model_path


def compare_training_runs(data_file: str, generations: list = [1, 2], epochs: int = 5):
    """Train multiple generations on the same data for comparison."""
    print(f"🏁 Comparative training: Generations {generations}")
    print(f"📊 Using same data: {data_file}")

    trained_models = {}

    for gen in generations:
        print(f"\n{'=' * 50}")
        print(f"Training Generation {gen}")
        print(f"{'=' * 50}")

        model_path = train_model(
            data_file=data_file,
            generation=gen,
            epochs=epochs,
            output_name=f"comparison_gen{gen}_{datetime.now().strftime('%H%M%S')}",
        )

        trained_models[gen] = model_path

    print(f"\n🎯 Comparative training complete!")
    print("Trained models:")
    for gen, path in trained_models.items():
        print(f"  Generation {gen}: {path}")

    return trained_models


def incremental_training(
    base_model_path: str, new_data_file: str, epochs: int = 5, output_name: str = None
) -> str:
    """Continue training an existing model with new data."""

    print(f"🔄 Incremental training from {base_model_path}")

    # Load new training data
    with open(new_data_file, "rb") as f:
        data = pickle.load(f)

    training_positions = data["positions"]
    generation = data["generation"]

    print(f"📖 Loaded {len(training_positions)} new positions")

    # Create output name if not provided
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"incremental_{timestamp}"

    # Load existing model
    ai = ChessAI(
        model_path=base_model_path, generation=generation, run_name=output_name
    )

    print(f"✅ Loaded existing model")
    print(f"🚀 Continuing training for {epochs} epochs...")

    # Continue training
    ai.train_on_data(training_data=training_positions, epochs=epochs, batch_size=64)

    # Save updated model
    model_path = f"{ai.run_dir}/incremental_model.pth"
    ai.save_model(model_path)

    print(f"💾 Updated model saved to {model_path}")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train chess AI models")
    parser.add_argument("data_file", help="Training data file (.pkl)")
    parser.add_argument(
        "--generation",
        "-g",
        type=int,
        default=1,
        choices=[1, 2],
        help="Model generation (default: 1)",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=10, help="Training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output model name (default: auto-generated)"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Train both generations for comparison"
    )
    parser.add_argument(
        "--incremental",
        metavar="BASE_MODEL",
        help="Continue training from existing model",
    )

    args = parser.parse_args()

    # Check if data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Data file not found: {args.data_file}")
        print("💡 Generate training data first with: uv run python generate_data.py")
        return

    if args.incremental:
        # Incremental training
        if not Path(args.incremental).exists():
            print(f"❌ Base model not found: {args.incremental}")
            return

        incremental_training(
            base_model_path=args.incremental,
            new_data_file=args.data_file,
            epochs=args.epochs,
            output_name=args.output,
        )

    elif args.compare:
        # Comparative training
        compare_training_runs(
            data_file=args.data_file, generations=[1, 2], epochs=args.epochs
        )

    else:
        # Standard training
        train_model(
            data_file=args.data_file,
            generation=args.generation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            output_name=args.output,
        )


if __name__ == "__main__":
    main()
