from pathlib import Path
from src.preprocessing.base import BaseFeatureProcessor

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


class SonicsPreprocessor(BaseFeatureProcessor):
    def __init__(self, input_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.input_dir = Path(input_dir)

    def collect_files(self) -> list[tuple[Path, str]]:
        files = [
            f for f in self.input_dir.rglob("*")
            if f.suffix.lower() in AUDIO_EXTENSIONS
        ]
        return [(f, f.stem) for f in files]
