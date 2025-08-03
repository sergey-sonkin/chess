# Chess AI

Simple chess neural net because I was feeling nostalgic.

## File structure

- benchmark.py - Performance testing (CPU v MPS v CUDA)
- generate_data.py - Pure game generation
- play.py - Human vs AI interface
- tournaments.py - AI vs AI battles
- train_model.py - Model training from saved data


## Running

1. Generate Training Data (once):
```
uv run python generate_data.py --games 5000 --generation 1
```

2. Train Models (reusing data):
```
uv run python train_model.py data/training_data_*.pkl --generation 1
uv run python train_model.py data/training_data_*.pkl --generation 2 --compare
```

3. Test Performance:
```
uv run python benchmark.py
```

4. Battle Models:
```
uv run python tournaments.py --model1 path1.pth --model2 path2.pth
```

5. Play Against AI:
```
uv run python play.py --generation 2
```

