# Chess AI

Simple chess neural net because I was feeling nostalgic.

## File structure

- benchmark.py - Performance testing (CPU v MPS v CUDA)
- generate_data.py - Pure game generation
- play.py - Human vs AI interface
- tournaments.py - AI vs AI battles
- train_model.py - Model training from saved data


## Running

* Training the models: `uv run python chess_ai.py`
* Benchmarking CPU v Apple ML cores: `uv run python benchmark.py`
