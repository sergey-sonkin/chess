#!/usr/bin/env python3
"""
Human vs AI chess interface.
Load trained models and play against them.
"""

import argparse
import os
from pathlib import Path

from chess_ai import ChessAI


def find_latest_model(generation: int | None = None) -> str | None:
    """Find the most recent trained model."""
    if not os.path.exists("runs"):
        return None

    runs = [d for d in os.listdir("runs") if os.path.isdir(f"runs/{d}")]
    if not runs:
        return None

    # Filter by generation if specified
    if generation:
        runs = [r for r in runs if f"gen{generation}" in r]
        if not runs:
            return None

    # Get most recent run
    latest_run = max(runs)
    run_dir = f"runs/{latest_run}"

    # Look for models in the run directory
    model_files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]
    if not model_files:
        return None

    # Prefer final_model.pth, otherwise get the latest iteration
    if "final_model.pth" in model_files:
        return f"{run_dir}/final_model.pth"
    else:
        latest_model = max(model_files)
        return f"{run_dir}/{latest_model}"


def play_against_ai(model_path: str | None = None, generation: int | None = None):
    """Play a game against the trained AI."""

    if model_path is None:
        model_path = find_latest_model(generation)
        if model_path is None:
            print("❌ No trained models found!")
            print(
                "💡 Train a model first with: uv run python train_model.py data/your_data.pkl"
            )
            return
        print(f"🤖 Using latest model: {model_path}")
    elif not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Load the AI
    try:
        ai = ChessAI(model_path=model_path, generation=generation or 1)
        print("✅ Loaded AI model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    import chess

    board = chess.Board()

    print("\n♟️  Chess AI Game Started!")
    print("📝 Enter moves in algebraic notation (e.g., e4, Nf3, O-O)")
    print(
        "💡 Commands: 'quit' to exit, 'board' to show position, 'eval' for AI evaluation"
    )
    print("🎯 You are White, AI is Black")
    print("\n" + "=" * 50)

    while not board.is_game_over():
        print(f"\n{board}")
        print(f"Turn: {'White (You)' if board.turn else 'Black (AI)'}")

        if board.turn:  # Human plays white
            while True:
                try:
                    move_input = input("Your move: ").strip().lower()

                    if move_input == "quit":
                        print("👋 Thanks for playing!")
                        return
                    elif move_input == "board":
                        print(f"\n{board}")
                        continue
                    elif move_input == "eval":
                        eval_score = ai.evaluate_position(board)
                        print(f"🧠 AI evaluation: {eval_score:.3f}")
                        print(
                            "   (Positive = good for White, Negative = good for Black)"
                        )
                        continue

                    move = board.parse_san(move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("❌ Illegal move! Try again.")
                except ValueError:
                    print(
                        "❌ Invalid move format! Use algebraic notation (e.g., e4, Nf3)"
                    )
                except KeyboardInterrupt:
                    print("\n👋 Thanks for playing!")
                    return

        else:  # AI plays black
            print("🤔 AI is thinking...")
            ai_move = ai.select_move(
                board, exploration_rate=0.0
            )  # No exploration in actual play
            board.push(ai_move)
            print(f"🤖 AI plays: {ai_move}")

            # Show AI's evaluation of the new position
            eval_score = ai.evaluate_position(board)
            print(f"🧠 AI evaluation: {eval_score:.3f}")

    # Game over
    print(f"\n{board}")
    print("\n🏁 Game Over!")
    result = board.result()
    print(f"📊 Result: {result}")

    if board.is_checkmate():
        if result == "1-0":
            print("🎉 Congratulations! You won by checkmate!")
        else:
            print("🤖 AI wins by checkmate!")
    elif board.is_stalemate():
        print("🤝 Draw by stalemate!")
    elif board.is_insufficient_material():
        print("🤝 Draw by insufficient material!")
    elif board.is_seventyfive_moves():
        print("🤝 Draw by 75-move rule!")
    elif board.is_fivefold_repetition():
        print("🤝 Draw by repetition!")


def list_available_models():
    """List all available trained models."""
    if not os.path.exists("runs"):
        print("❌ No runs directory found")
        return

    runs = [d for d in os.listdir("runs") if os.path.isdir(f"runs/{d}")]
    if not runs:
        print("❌ No training runs found")
        return

    print("🤖 Available Models:")
    print("=" * 50)

    for run in sorted(runs):
        run_dir = f"runs/{run}"
        model_files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]

        if model_files:
            print(f"\n📁 {run}")
            for model_file in sorted(model_files):
                model_path = f"{run_dir}/{model_file}"
                print(f"   └── {model_file}")

                # Try to load metadata if available
                metadata_path = f"{run_dir}/training_metadata.pkl"
                if os.path.exists(metadata_path):
                    try:
                        import pickle

                        with open(metadata_path, "rb") as f:
                            metadata = pickle.load(f)
                        print(
                            f"       Generation: {metadata.get('generation', 'Unknown')}"
                        )
                        print(f"       Epochs: {metadata.get('epochs', 'Unknown')}")
                        print(
                            f"       Positions: {metadata.get('num_training_positions', 'Unknown')}"
                        )
                    except:
                        pass


def main():
    parser = argparse.ArgumentParser(description="Play chess against trained AI")
    parser.add_argument("--model", "-m", type=str, help="Path to specific model file")
    parser.add_argument(
        "--generation",
        "-g",
        type=int,
        choices=[1, 2],
        help="Prefer models from specific generation",
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available models"
    )

    args = parser.parse_args()

    if args.list:
        list_available_models()
    else:
        play_against_ai(model_path=args.model, generation=args.generation)


if __name__ == "__main__":
    main()
