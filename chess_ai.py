import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

from torch.types import Tensor


@dataclass
class GamePosition:
    """Store a chess position with its eventual outcome."""

    board_tensor: torch.Tensor
    outcome: float  # Reward value from perspective of player to move


def reward_function_gen1(result: str, game_length: int, max_length: int = 200) -> float:
    """Generation 1 reward function: Simple win/loss/draw."""
    if result == "1-0":  # White wins
        return 1.0
    elif result == "0-1":  # Black wins
        return -1.0
    else:  # Draw
        return 0.0


def reward_function_gen2(result: str, game_length: int, max_length: int = 200) -> float:
    """Generation 2 reward function: Length-aware rewards."""
    length_factor = (max_length - game_length) / max_length * 0.5

    if result == "1-0":  # White wins
        return 1.0 + length_factor  # Bonus for shorter wins
    elif result == "0-1":  # Black wins
        return -1.0 - length_factor  # Penalty increases for shorter losses
    else:  # Draw
        return -0.1  # Small penalty to encourage decisive play


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert chess board to tensor representation."""
    tensor = torch.zeros(8, 8, 12)

    piece_map = {
        chess.PAWN: 0,
        chess.ROOK: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        piece_type = piece_map[piece.piece_type]
        color_offset = 0 if piece.color else 6
        tensor[row, col, piece_type + color_offset] = 1

    return tensor.flatten()


class ChessNet(nn.Module):
    """Neural network for chess position evaluation."""

    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(768, 512)  # 8*8*12 = 768
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x


class ChessAI:
    """Chess AI that learns through self-play."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        run_name: Optional[str] = None,
        generation: int = 1,
        reward_function=None,
    ):
        self.model = ChessNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.generation = generation

        # Set reward function based on generation
        if reward_function is None:
            if generation == 1:
                self.reward_function = reward_function_gen1
            elif generation == 2:
                self.reward_function = reward_function_gen2
            else:
                self.reward_function = reward_function_gen1  # Default to gen1
        else:
            self.reward_function = reward_function

        # Set up run directory
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"gen{generation}_run_{timestamp}"
        else:
            self.run_name = run_name

        self.run_dir = f"runs/{self.run_name}"
        os.makedirs(self.run_dir, exist_ok=True)

        if model_path and self.load_model(model_path):
            print(f"Loaded model from {model_path}")
        else:
            print(f"Starting fresh Generation {generation} model in {self.run_dir}")

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a chess position using the neural network."""
        self.model.eval()
        with torch.no_grad():
            board_tensor = board_to_tensor(board).unsqueeze(0)
            evaluation = self.model(board_tensor).item()
        return evaluation

    def select_move(
        self, board: chess.Board, exploration_rate: float = 0.1
    ) -> chess.Move:
        """Select move using network evaluation with some exploration."""
        legal_moves = list(board.legal_moves)

        if random.random() < exploration_rate:
            return random.choice(legal_moves)

        best_move = None
        best_value = float("-inf") if board.turn else float("inf")

        for move in legal_moves:
            board.push(move)
            value = self.evaluate_position(board)
            board.pop()

            if board.turn and value > best_value:
                best_value = value
                best_move = move
            elif not board.turn and value < best_value:
                best_value = value
                best_move = move

        return best_move or random.choice(legal_moves)

    def play_self_game(self, exploration_rate: float = 0.3) -> List[GamePosition]:
        """Play a game against itself and collect position data."""
        board = chess.Board()
        positions: list[tuple[Tensor, chess.Color]] = []

        while not board.is_game_over() and len(positions) < 200:  # Limit game length
            board_tensor = board_to_tensor(board)
            positions.append((board_tensor.clone(), board.turn))

            move = self.select_move(board, exploration_rate)
            board.push(move)

        if not board.is_game_over() and len(positions) >= 200:
            print("Game didn't finish after 200 moves")

        # Use generational reward function
        result = board.result()
        game_length = len(positions)
        white_outcome = self.reward_function(result, game_length)

        # Create training data with outcomes
        game_positions = []
        for board_tensor, was_white_turn in positions:
            # Flip outcome if it was black's turn
            position_outcome = white_outcome if was_white_turn else -white_outcome
            game_positions.append(GamePosition(board_tensor, position_outcome))

        return game_positions

    def generate_training_data(
        self, num_games: int, exploration_rate: float = 0.3
    ) -> List[GamePosition]:
        """Generate training data through self-play."""
        all_positions = []

        print(f"Generating training data from {num_games} self-play games...")
        for game_num in range(num_games):
            print(f"Just completed Game {game_num}")
            if (game_num + 1) % 100 == 0:
                print(f"Game {game_num + 1}/{num_games}")

            game_positions = self.play_self_game(exploration_rate)
            all_positions.extend(game_positions)

        print(f"Generated {len(all_positions)} training positions")
        return all_positions

    def train_on_data(
        self, training_data: List[GamePosition], epochs: int = 10, batch_size: int = 64
    ):
        """Train the neural network on position data."""
        self.model.train()

        # Convert to tensors
        positions = torch.stack([pos.board_tensor for pos in training_data])
        outcomes = torch.tensor(
            [pos.outcome for pos in training_data], dtype=torch.float32
        ).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(positions, outcomes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        print(f"Training on {len(training_data)} positions for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_positions, batch_outcomes in dataloader:
                self.optimizer.zero_grad()

                predictions = self.model(batch_positions)
                loss = self.criterion(predictions, batch_outcomes)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def iterative_training(self, iterations: int = 5, games_per_iteration: int = 500):
        """Perform iterative self-play training."""
        print("Starting iterative self-play training...")

        for iteration in range(iterations):
            print(f"\n=== Iteration {iteration + 1}/{iterations} ===")

            # Decrease exploration over time
            exploration_rate = 0.5 * (0.8**iteration)
            print(f"Exploration rate: {exploration_rate:.3f}")

            # Generate training data
            training_data = self.generate_training_data(
                games_per_iteration, exploration_rate
            )

            # Train on the data
            self.train_on_data(training_data, epochs=5)

            # Save model checkpoint with generation info
            model_path = f"{self.run_dir}/gen{self.generation}_iter_{iteration + 1}.pth"
            self.save_model(model_path)
            print(f"Generation {self.generation} model saved as {model_path}")

    def save_model(self, path: str):
        """Save the trained model."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str) -> bool:
        """Load a trained model."""
        try:
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return True
        except FileNotFoundError:
            return False


def play_against_ai(model_path: Optional[str] = None):
    """Play a game against the trained AI."""
    if model_path is None:
        # Find latest model in most recent run
        if os.path.exists("runs"):
            runs = [d for d in os.listdir("runs") if os.path.isdir(f"runs/{d}")]
            if runs:
                latest_run = max(runs)
                run_models = [
                    f for f in os.listdir(f"runs/{latest_run}") if f.endswith(".pth")
                ]
                if run_models:
                    latest_model = max(run_models)
                    model_path = f"runs/{latest_run}/{latest_model}"
                    print(f"Using latest model: {model_path}")

    ai = ChessAI(model_path)
    board = chess.Board()

    print("Chess AI Game Started!")
    print("Enter moves in algebraic notation (e.g., e4, Nf3, O-O)")
    print("Type 'quit' to exit\n")

    while not board.is_game_over():
        print(f"\n{board}")
        print(f"Turn: {'White (You)' if board.turn else 'Black (AI)'}")

        if board.turn:  # Human plays white
            while True:
                try:
                    move_input = input("Your move: ").strip()
                    if move_input.lower() == "quit":
                        return

                    move = board.parse_san(move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move! Try again.")
                except ValueError:
                    print("Invalid move format! Use algebraic notation (e.g., e4, Nf3)")
        else:  # AI plays black
            print("AI is thinking...")
            ai_move = ai.select_move(
                board, exploration_rate=0.0
            )  # No exploration in actual play
            board.push(ai_move)
            print(f"AI plays: {ai_move}")

            # Show AI's evaluation
            eval_score = ai.evaluate_position(board)
            print(f"AI evaluation: {eval_score:.3f}")

    print(f"\nGame Over!")
    print(f"Result: {board.result()}")


if __name__ == "__main__":
    # Train Generation 1 AI (simple rewards)
    print("=== Training Generation 1 (Simple Rewards) ===")
    gen1_ai = ChessAI(generation=1)
    gen1_ai.iterative_training(iterations=3, games_per_iteration=200)

    # Train Generation 2 AI (length-aware rewards)
    print("\n=== Training Generation 2 (Length-Aware Rewards) ===")
    gen2_ai = ChessAI(generation=2)
    gen2_ai.iterative_training(iterations=3, games_per_iteration=200)

    # Play against latest generation
    print("\n" + "=" * 50)
    print("Training complete! You can now play against the latest AI.")
    play_against_ai()
