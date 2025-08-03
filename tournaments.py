import chess
from dataclasses import dataclass
from typing import Optional
from chess_ai import ChessAI


@dataclass
class TournamentResult:
    """Store results of a tournament between two AIs."""

    player1_name: str
    player2_name: str
    player1_wins: int
    player2_wins: int
    draws: int
    total_games: int
    avg_game_length: float

    @property
    def player1_winrate(self) -> float:
        return self.player1_wins / self.total_games

    @property
    def player2_winrate(self) -> float:
        return self.player2_wins / self.total_games

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games


def play_ai_vs_ai_game(ai1: ChessAI, ai2: ChessAI, show_progress: bool = False) -> str:
    """Play a single game between two AIs and return the result."""
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < 200:
        current_ai = ai1 if board.turn else ai2
        move = current_ai.select_move(
            board, exploration_rate=0.0
        )  # No exploration in tournaments
        board.push(move)
        move_count += 1

        if show_progress and move_count % 50 == 0:
            print(f"Move {move_count}...")

    return board.result()


def tournament(
    ai1: ChessAI,
    ai2: ChessAI,
    num_games: int = 100,
    ai1_name: str = "AI1",
    ai2_name: str = "AI2",
) -> TournamentResult:
    """Run a tournament between two AIs."""
    print(f"\n🏆 Tournament: {ai1_name} vs {ai2_name}")
    print(f"Playing {num_games} games...")

    ai1_wins = 0
    ai2_wins = 0
    draws = 0
    total_moves = 0

    for game_num in range(num_games):
        if (game_num + 1) % 20 == 0:
            print(f"Game {game_num + 1}/{num_games}")

        # Alternate who plays white to be fair
        if game_num % 2 == 0:
            result = play_ai_vs_ai_game(ai1, ai2)
        else:
            result = play_ai_vs_ai_game(ai2, ai1)
            # Flip result since we swapped colors
            if result == "1-0":
                result = "0-1"
            elif result == "0-1":
                result = "1-0"

        # Count wins from AI1's perspective
        if result == "1-0" and game_num % 2 == 0:  # AI1 as white wins
            ai1_wins += 1
        elif result == "0-1" and game_num % 2 == 1:  # AI1 as black wins
            ai1_wins += 1
        elif result == "1-0" and game_num % 2 == 1:  # AI2 as white wins
            ai2_wins += 1
        elif result == "0-1" and game_num % 2 == 0:  # AI2 as black wins
            ai2_wins += 1
        else:  # Draw
            draws += 1

    avg_game_length = total_moves / num_games if num_games > 0 else 0

    result = TournamentResult(
        player1_name=ai1_name,
        player2_name=ai2_name,
        player1_wins=ai1_wins,
        player2_wins=ai2_wins,
        draws=draws,
        total_games=num_games,
        avg_game_length=avg_game_length,
    )

    # Print results
    print(f"\n📊 Tournament Results:")
    print(f"{ai1_name}: {ai1_wins} wins ({result.player1_winrate:.1%})")
    print(f"{ai2_name}: {ai2_wins} wins ({result.player2_winrate:.1%})")
    print(f"Draws: {draws} ({result.draw_rate:.1%})")
    print(f"Total games: {num_games}")

    return result


def compare_generations(
    gen1_model_path: str, gen2_model_path: str, num_games: int = 50
):
    """Compare two generation models in a tournament."""
    print("Loading models for generation comparison...")

    gen1_ai = ChessAI(model_path=gen1_model_path, generation=1)
    gen2_ai = ChessAI(model_path=gen2_model_path, generation=2)

    result = tournament(
        gen1_ai,
        gen2_ai,
        num_games=num_games,
        ai1_name="Generation 1",
        ai2_name="Generation 2",
    )

    # Determine winner
    if result.player1_wins > result.player2_wins:
        print(f"\n🏅 Generation 1 wins the tournament!")
    elif result.player2_wins > result.player1_wins:
        print(f"\n🏅 Generation 2 wins the tournament!")
    else:
        print(f"\n🤝 Tournament ends in a tie!")

    return result
