# SVDD
Singing Voice Deepfake Detection

### Seting up env:
1. Create `uv` env
2. Use `uv sync` to download all packages with proper versions based on `uv.lock` file

### Download data
1. To dowload data you can use prepared scripts under `scripts/` directory

### Preprocess data
1. To preprocess downloaded datasets you can use scripts under `scripts/processing/`. Right now scripts are prepared ONLY for SingFake dataset.
2. First: `simulate_codec_singfake.py` -> `demucs_vad_singfake.py` -> `split_singfake.py`
3. It will allow you to create data that matches dataloaders and created datasets.



