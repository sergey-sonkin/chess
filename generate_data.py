#!/usr/bin/env python3
"""
Generate training data through self-play games.
Saves position-outcome pairs to files for later training.
"""

import argparse
import pickle
import random
from pathlib import Path
from datetime import datetime
from chess_ai import ChessAI, GamePosition


def generate_training_data(
    num_games: int,
    exploration_rate: float = 0.3,
    output_file: str = None,
    generation: int = 1,
    device: str = None,
) -> str:
    """Generate training data and save to file."""

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"training_data_{timestamp}.pkl"

    print(f"🎲 Generating {num_games} self-play games...")
    print(f"📊 Using Generation {generation} reward function")
    print(f"🎯 Exploration rate: {exploration_rate}")

    # Create AI for game generation (no model loading, just for game logic)
    ai = ChessAI(generation=generation, device=device)

    all_positions = []
    games_completed = 0

    for game_num in range(num_games):
        if (game_num + 1) % 100 == 0:
            print(f"Game {game_num + 1}/{num_games}")

        try:
            game_positions = ai.play_self_game(exploration_rate)
            all_positions.extend(game_positions)
            games_completed += 1
        except Exception as e:
            print(f"Error in game {game_num + 1}: {e}")
            continue

    print(f"✅ Generated {len(all_positions)} positions from {games_completed} games")

    # Save data
    data = {
        "positions": all_positions,
        "generation": generation,
        "num_games": games_completed,
        "exploration_rate": exploration_rate,
        "timestamp": datetime.now().isoformat(),
    }

    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    full_path = f"data/{output_file}"

    with open(full_path, "wb") as f:
        pickle.dump(data, f)

    print(f"💾 Saved training data to {full_path}")
    print(f"📈 Positions per game: {len(all_positions) / games_completed:.1f}")

    return full_path


def load_training_data(file_path: str):
    """Load training data from file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"📖 Loaded {len(data['positions'])} positions")
    print(f"🎮 From {data['num_games']} games")
    print(f"📅 Generated: {data['timestamp']}")
    print(f"🧬 Generation: {data['generation']}")

    return data


def combine_datasets(*file_paths: str, output_file: str = None):
    """Combine multiple training datasets into one."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"combined_data_{timestamp}.pkl"

    all_positions = []
    total_games = 0
    generations = set()

    for file_path in file_paths:
        data = load_training_data(file_path)
        all_positions.extend(data["positions"])
        total_games += data["num_games"]
        generations.add(data["generation"])

    combined_data = {
        "positions": all_positions,
        "generation": list(generations) if len(generations) == 1 else "mixed",
        "num_games": total_games,
        "exploration_rate": "mixed",
        "timestamp": datetime.now().isoformat(),
        "source_files": file_paths,
    }

    full_path = f"data/{output_file}"
    with open(full_path, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"🔗 Combined {len(file_paths)} datasets")
    print(f"📊 Total positions: {len(all_positions)}")
    print(f"💾 Saved to {full_path}")

    return full_path


def main():
    parser = argparse.ArgumentParser(description="Generate chess training data")
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=1000,
        help="Number of games to generate (default: 1000)",
    )
    parser.add_argument(
        "--exploration",
        "-e",
        type=float,
        default=0.3,
        help="Exploration rate (default: 0.3)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output filename (default: auto-generated)"
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=1,
        choices=[1, 2],
        help="Reward function generation (default: 1)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--combine", nargs="+", metavar="FILE", help="Combine existing data files"
    )

    args = parser.parse_args()

    if args.combine:
        combine_datasets(*args.combine, output_file=args.output)
    else:
        generate_training_data(
            num_games=args.games,
            exploration_rate=args.exploration,
            output_file=args.output,
            generation=args.generation,
            device=args.device,
        )


if __name__ == "__main__":
    main()
