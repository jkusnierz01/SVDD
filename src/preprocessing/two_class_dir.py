from pathlib import Path
from src.preprocessing.base import BaseFeatureProcessor


class TwoClassDirPreprocessor(BaseFeatureProcessor):
    """Preprocessor for datasets organised as two directories: deepfake and bonafide.

    Stems are prefixed with the directory name to avoid collisions and to
    encode class membership, e.g.:
      ai/foo.wav   -> stem "ai_foo"
      human/bar.wav -> stem "human_bar"
    """

    def __init__(
        self,
        deepfake_dir: str,
        bonafide_dir: str,
        extensions: list[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.deepfake_dir = Path(deepfake_dir)
        self.bonafide_dir = Path(bonafide_dir)
        self.extensions = {ext.lower() for ext in extensions}

    def collect_files(self) -> list[tuple[Path, str]]:
        files = []
        for directory in (self.deepfake_dir, self.bonafide_dir):
            prefix = directory.name  # "ai", "human", "DF", "BF", etc.
            for f in directory.rglob("*"):
                if f.suffix.lower() in self.extensions:
                    files.append((f, f"{prefix}_{f.stem}"))
        return files
