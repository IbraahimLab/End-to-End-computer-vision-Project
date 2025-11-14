from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
    zip_file_path: Path
    unzip_dir: Path



