import time
import torch
from chess_ai import ChessAI, GamePosition, board_to_tensor
import chess


def benchmark_device_performance(device: str, num_games: int = 10, num_epochs: int = 3):
    """Benchmark training performance on a specific device."""
    print(f"\n🚀 Benchmarking {device.upper()} performance...")
    print(f"Running {num_games} self-play games + {num_epochs} training epochs")

    start_time = time.time()

    # Create AI with specific device
    ai = ChessAI(generation=1, device=device, run_name=f"benchmark_{device}")

    # Benchmark self-play data generation
    print(f"Generating training data on {device}...")
    data_gen_start = time.time()
    training_data = ai.generate_training_data(num_games, exploration_rate=0.3)
    data_gen_time = time.time() - data_gen_start

    # Benchmark training
    print(f"Training neural network on {device}...")
    training_start = time.time()
    ai.train_on_data(training_data, epochs=num_epochs, batch_size=32)
    training_time = time.time() - training_start

    total_time = time.time() - start_time

    return {
        "device": device,
        "data_generation_time": data_gen_time,
        "training_time": training_time,
        "total_time": total_time,
        "positions_generated": len(training_data),
        "positions_per_second": len(training_data) / data_gen_time
        if data_gen_time > 0
        else 0,
    }


def benchmark_inference_speed(device: str, num_evaluations: int = 1000):
    """Benchmark neural network inference speed."""
    print(f"\n⚡ Benchmarking {device.upper()} inference speed...")
    print(f"Running {num_evaluations} position evaluations")

    # Create AI and a test board
    ai = ChessAI(generation=1, device=device)
    board = chess.Board()

    # Warm up
    for _ in range(10):
        ai.evaluate_position(board)

    # Benchmark inference
    start_time = time.time()
    for _ in range(num_evaluations):
        ai.evaluate_position(board)
    inference_time = time.time() - start_time

    return {
        "device": device,
        "inference_time": inference_time,
        "evaluations_per_second": num_evaluations / inference_time,
    }


def run_full_benchmark():
    """Run comprehensive benchmarks comparing CPU vs MPS."""
    print("🎯 Chess AI Device Performance Benchmark")
    print("=" * 50)

    devices_to_test = ["cpu"]

    # Check if MPS is available
    if torch.backends.mps.is_available():
        devices_to_test.append("mps")
        print("✅ MPS (Apple Silicon) detected")
    else:
        print("❌ MPS not available")

    # Check if CUDA is available
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
        print("✅ CUDA detected")
    else:
        print("❌ CUDA not available")

    print(f"Testing devices: {', '.join(devices_to_test)}")

    # Training benchmarks
    training_results = []
    for device in devices_to_test:
        try:
            result = benchmark_device_performance(device, num_games=20, num_epochs=3)
            training_results.append(result)
        except Exception as e:
            print(f"❌ Error benchmarking {device}: {e}")

    # Inference benchmarks
    inference_results = []
    for device in devices_to_test:
        try:
            result = benchmark_inference_speed(device, num_evaluations=500)
            inference_results.append(result)
        except Exception as e:
            print(f"❌ Error benchmarking inference on {device}: {e}")

    # Print results
    print("\n" + "=" * 50)
    print("📊 TRAINING BENCHMARK RESULTS")
    print("=" * 50)

    if len(training_results) > 1:
        baseline = next(r for r in training_results if r["device"] == "cpu")

        for result in training_results:
            device = result["device"].upper()
            total_time = result["total_time"]
            training_time = result["training_time"]
            positions = result["positions_generated"]

            if result["device"] != "cpu":
                speedup = baseline["total_time"] / total_time
                training_speedup = baseline["training_time"] / training_time
                print(f"\n{device}:")
                print(
                    f"  Total time: {total_time:.2f}s ({speedup:.2f}x faster than CPU)"
                )
                print(
                    f"  Training time: {training_time:.2f}s ({training_speedup:.2f}x faster)"
                )
                print(f"  Positions: {positions}")
            else:
                print(f"\n{device} (baseline):")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  Training time: {training_time:.2f}s")
                print(f"  Positions: {positions}")

    print("\n" + "=" * 50)
    print("⚡ INFERENCE BENCHMARK RESULTS")
    print("=" * 50)

    if len(inference_results) > 1:
        baseline = next(r for r in inference_results if r["device"] == "cpu")

        for result in inference_results:
            device = result["device"].upper()
            eval_per_sec = result["evaluations_per_second"]

            if result["device"] != "cpu":
                speedup = eval_per_sec / baseline["evaluations_per_second"]
                print(f"\n{device}:")
                print(
                    f"  Evaluations/sec: {eval_per_sec:.1f} ({speedup:.2f}x faster than CPU)"
                )
            else:
                print(f"\n{device} (baseline):")
                print(f"  Evaluations/sec: {eval_per_sec:.1f}")

    # Summary
    if len(training_results) > 1 and "mps" in [r["device"] for r in training_results]:
        mps_result = next(r for r in training_results if r["device"] == "mps")
        cpu_result = next(r for r in training_results if r["device"] == "cpu")
        overall_speedup = cpu_result["total_time"] / mps_result["total_time"]

        print(f"\n🏆 SUMMARY")
        print(f"Overall MPS speedup: {overall_speedup:.2f}x faster than CPU")

        if overall_speedup > 1.5:
            print("✅ Significant performance improvement with Apple Silicon!")
        elif overall_speedup > 1.1:
            print("✅ Moderate performance improvement with Apple Silicon")
        else:
            print("⚠️  Minimal performance difference - CPU might be sufficient")


if __name__ == "__main__":
    run_full_benchmark()
